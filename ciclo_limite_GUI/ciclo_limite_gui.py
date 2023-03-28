import cv2
import numpy as np
import time
import paho.mqtt.client as mqtt

########### MQTT Parameters ###########
broker_hostname = "mosquitto"
broker_port = 1883
topic = "ciclo-limite"

########### Other parameters ###########
image_path = "../../sd-output/ciclo_limite/image_00000.png"
screen_width = 1920 # resolution in px
screen_height = 1080 # resolution in px

########### Global Parameters ###########
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


    # Create a black image with the same resolution as the screen
    img_black = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Create a full screen window with OpenCV
    cv2.namedWindow("Full Screen Image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Full Screen Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while 1:
        if show_image:
            show_image = False
            # Read the image and show it
            img = cv2.imread(image_path)
            cv2.imshow("Full Screen Image", img)

            # Wait for any key to be pressed and then close the window
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        time.sleep(1000)