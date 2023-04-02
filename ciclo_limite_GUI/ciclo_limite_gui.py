import cv2
import numpy as np
import time
import paho.mqtt.client as mqtt
from deep_translator import GoogleTranslator
import random
from PIL import ImageFont, ImageDraw, Image


########### GUI Parameters ###########
min_sec = 10 # minimum number of seconds showing the image or text (must be different than zero)

# Image parameters #
background_color = (255, 0, 0)  # BGR format (Blue Green Red format, from 0 to 255)

# Text parameters #
custom_font = True
if not custom_font:
    # This is by using OpenCV, font cannot be changed
    text_color = (255,255,255) # BGR format (Blue Green Red format, from 0 to 255) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    thickness = 6
    max_text_size = 1800 # in pixels
    offset_line_spacing = 40 # in pixels
else:
    # Custom font, Pillow library must be used. Parameters differs from OpenCV method
    font_path = "RockfordSans-Regular.otf"
    font_size = 100
    text_color = (255,255,255) # RGB format
    offset_line_spacing = 25 # in pixels
    max_text_size = 1800 # in pixels

########### MQTT Parameters ###########
broker_hostname = "172.18.0.1"
broker_port = 1883
topic = "ciclo-limite"

########### Other parameters ###########
image_path = "../../sd-output/ciclo_limite/image_00000.png"
text_path = "../../sd-output/ciclo_limite/caption.txt"
screen_width = 1920 # resolution in px
screen_height = 1080 # resolution in px

########### Global Variables ###########
show_image = False
show_text = False
new_prompt = ""



# Define a callback function for when a message is received
def on_message(client, userdata, message):
    global show_image, show_text, new_prompt
    
    print("Received message: ", str(message.payload.decode("utf-8")))

    if message.payload.decode("utf-8") == "sd_unlock":
        show_text = True
        show_image = False
    elif message.payload.decode("utf-8") == "vit_gpt2_unlock":
        show_text = False
        show_image = True
    else:
        new_prompt = str(message.payload.decode("utf-8"))
        show_text = True
        show_image = False

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
    else:
        print("Connection failed. Retrying in 5 seconds...")
        time.sleep(5)
        client.connect(broker_address)



def crop_text(text, size):
    subtexts = []
    current_subtext = ""

    for word in text.split():
        current_subtext_size, _ = cv2.getTextSize(current_subtext, font, font_scale, thickness)
        word_size, _ = cv2.getTextSize(word, font, font_scale, thickness)
        if current_subtext_size[0] + word_size[0] > size:
            subtexts.append(current_subtext)
            current_subtext = ""
        current_subtext += " " + word

    if current_subtext:
        subtexts.append(current_subtext.strip())

    return subtexts

def crop_text_pil(text, size, font):
    subtexts = []
    current_subtext = ""

    for word in text.split():
        current_subtext_width, current_subtext_height = font.getsize(current_subtext)
        current_subtext_size = (current_subtext_width,current_subtext_height)
        word_width, word_height = font.getsize(word)
        word_size = (word_width,word_height)
        if current_subtext_size[0] + word_size[0] > size:
            subtexts.append(current_subtext)
            current_subtext = ""
        current_subtext += " " + word

    if current_subtext:
        subtexts.append(current_subtext.strip())

    return subtexts

if __name__ == "__main__":
    # Set up the MQTT client
    client = mqtt.Client()
    # Set callback functions
    client.on_connect = on_connect
    client.on_message = on_message
    # Connect to the broker
    connected = False
    while not connected:
        try:
            client.connect(broker_hostname)
            connected = True
        except:
            print("Connection refused. Retrying in 1 seconds...")
            time.sleep(1)
    # Subscribe to a topic
    client.subscribe(topic)
    # Start the client loop
    client.loop_start()

    if custom_font:
        # Load the custom font
        font = ImageFont.truetype(font_path, font_size)

    # Create a background image with the same resolution as the screen
    background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    background[:] = background_color

    # Create a full screen window with OpenCV
    cv2.namedWindow("Full Screen Image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Full Screen Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while 1:
        # General time sleep
        time.sleep(1)
        if show_image:
            show_image = False

            # Read the image and show it
            img = cv2.imread(image_path)

            # Calculate the position to place the image in the center of the screen
            # NOTE: take into account that we want to show the image (512x512) as big as posible
            # at full screen. So, because screens are usually wider, we consider that the max height
            # is 512. Preserve 16:9 resolution.
            max_height = img.shape[0]
            max_width = int(screen_width/screen_height*max_height)
            dst = cv2.resize(background, (max_width, max_height))
            # Compute the position where the image will be
            x_pos = (max_width - img.shape[1]) // 2
            y_pos = (max_height - img.shape[0]) // 2
            # Display the image on top of the background
            dst[y_pos:y_pos+img.shape[0], x_pos:x_pos+img.shape[1]] = img

            # Show the current image on full screen
            cv2.imshow("Full Screen Image", dst)

            # Wait for a key press min_sec seconds and then continue with the loop
            cv2.waitKey(min_sec*1000)

        elif show_text:
            show_text = False

            if new_prompt:
                text = new_prompt
                new_prompt = ""
            else:
                # Open a file if exists and read the predicted caption
                with open(text_path, "r") as text_file:
                    text = text_file.readline()
            # print(text)

            # Translate prompt
            text = GoogleTranslator(source='auto', target='es').translate(text)

            # Remove word "montón" because it appears "un montón de veces"
            new_string = ""
            if "montón" in text:
                for word in text.split():
                    if word == "montón" and random.random() > 0.5:
                        new_string += "conjunto "
                    else:
                        new_string += word + " "
                text = new_string.strip()

            if custom_font:
                text_width, text_height = font.getsize(text)
                text_size = (text_width,text_height)
                text_x = (background.shape[1] - text_size[0]) // 2
                text_y = (background.shape[0] - text_size[1]) // 2
            else:
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = (background.shape[1] - text_size[0]) // 2
                text_y = (background.shape[0] + text_size[1]) // 2

            # Create an image with text
            if not custom_font:
                img = np.copy(background) # Copy the background in img

                # Check if the text exceeds the width of the image
                if text_size[0] > max_text_size:
                    subtexts = crop_text(text, max_text_size)

                    for i,text in enumerate(subtexts):
                        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                        text_x = (background.shape[1] - text_size[0]) // 2
                        text_y = (background.shape[0] + text_size[1]) // len(subtexts)
                        img = cv2.putText(img, text, (text_x, (text_y+offset_line_spacing)+(text_size[1]+offset_line_spacing)*i), font, font_scale, text_color, thickness, cv2.LINE_AA)
                else:
                    img = cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
            else:
                # Create an image with the desired text using the custom font
                img_pil = Image.new("RGB", (screen_width, screen_height), (background_color[2],background_color[1],background_color[0]))
                draw = ImageDraw.Draw(img_pil)

                # Check if the text exceeds the width of the image
                if text_size[0] > max_text_size:
                    subtexts = crop_text_pil(text, max_text_size, font)

                    for i,text in enumerate(subtexts):
                        text_width, text_height = font.getsize(text)
                        text_size = (text_width,text_height)
                        text_x = (background.shape[1] - text_size[0]) // 2
                        text_y = (background.shape[0] - text_size[1]) // len(subtexts)
                        draw.multiline_text((text_x, text_y+(text_size[1]+offset_line_spacing)*i), text, font=font, fill=text_color, spacing=offset_line_spacing, align="center")
                else:
                    draw.multiline_text((text_x, text_y), text, font=font, fill=text_color, spacing=offset_line_spacing, align="center")

                # Convert the PIL image to a NumPy array
                img_np = np.array(img_pil)
                # Convert the NumPy array to an OpenCV image
                img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Show the current image on full screen
            cv2.imshow("Full Screen Image", img)

            # Wait for a key press min_sec seconds and then continue with the loop
            cv2.waitKey(min_sec*1000)

    # Close the window
    cv2.destroyAllWindows()