import cv2
import numpy as np
import time
import paho.mqtt.client as mqtt

########### GUI Parameters ###########
min_sec = 10 # minimum number of seconds showing the image or text (must be different than zero)

# Text parameters #
text_color = (255,255,255) # BGR format (Blue Green Red format, from 0 to 255) 
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 4
thickness = 6
max_text_size = 1900 # in pixels
offset_line_spacing = 40 # in pixels

# Image parameters #
background_color = (255, 0, 0)  # BGR format (Blue Green Red format, from 0 to 255)

########### MQTT Parameters ###########
broker_hostname = "172.19.0.1"
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



# Define a callback function for when a message is received
def on_message(client, userdata, message):
    global show_image, show_text
    
    print("Received message: ", str(message.payload.decode("utf-8")))

    if message.payload.decode("utf-8") == "sd_unlock":
        show_text = True
        show_image = False
    elif message.payload.decode("utf-8") == "vit_gpt2_unlock":
        show_text = False
        show_image = True
    else:
        show_text = False
        show_image = False



def crop_text(text, size):
    subtexts = []
    current_subtext = ""

    for word in text.split():
        current_subtext_size, _ = cv2.getTextSize(current_subtext, font, font_scale, thickness)
        word_size, _ = cv2.getTextSize(current_subtext, font, font_scale, thickness)
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
    # Connect to the broker
    client.connect(broker_hostname, broker_port)
    # Subscribe to a topic
    client.subscribe(topic)
    # Set the callback function
    client.on_message = on_message
    # Start the client loop
    client.loop_start()


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
            max_width = int(16/9*max_height)
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
            # Open a file if exists and read the predicted caption
            with open(text_path, "r") as text_file:
                text = text_file.readline()
                # print(text)

            # Create an image with text
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (background.shape[1] - text_size[0]) // 2
            text_y = (background.shape[0] + text_size[1]) // 2

            img = np.copy(background) # Copy the background in img

            # Check if the text exceeds the width of the image
            if text_size[0] > max_text_size:
                subtexts = crop_text(text, max_text_size)

                for i,text in enumerate(subtexts):
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_x = (background.shape[1] - text_size[0]) // 2
                    text_y = (background.shape[0] + text_size[1]) // len(subtexts)
                    img = cv2.putText(img, text, (text_x, text_y+(text_size[1]+offset_line_spacing)*i), font, font_scale, text_color, thickness, cv2.LINE_AA)
            else:
                img = cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            # Show the current image on full screen
            cv2.imshow("Full Screen Image", img)

            # Wait for a key press min_sec seconds and then continue with the loop
            cv2.waitKey(min_sec*1000)

    # Close the window
    cv2.destroyAllWindows()