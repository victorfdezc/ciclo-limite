import urllib.request
import re
import time
import os
from threading import Thread
import paho.mqtt.client as mqtt

########### Ciclo Limite Parameters ###########
max_ncycles_prompt = 5 # max number of cycles per prompt
link = "https://rosmolarr.pythonanywhere.com/last"

########### Stable Diffusion Parameters ###########
ddim_steps = 15
scale = 7.5
n_iter = 1
n_samples =  1
outdir = "outputs/txt2img-samples"

########### VIT GPT2 Parameters ###########
max_length = 16
num_beams = 4

########### MQTT Parameters ###########
broker_hostname = "mosquitto"
broker_port = 1883
topic = "ciclo-limite"

############# Globals Definition #############
queue_prompt = []
vit_caption = ""
client = mqtt.Client() # Set up the MQTT client
sd_unlock = False
vit_gpt2_unlock = False
###############################################


def queue_thread():
    global queue_prompt
    prev_prompt = ""

    while 1:
        time.sleep(0.5)

        f = urllib.request.urlopen(link)
        txt = f.read().decode("utf-8")
        new_prompt = re.findall("<p>(.*?)<\/p>",txt)[0]

        if not new_prompt == prev_prompt:
            queue_prompt.append(new_prompt)
            print("New prompt: " + str(queue_prompt))
            prev_prompt = new_prompt

            # Publish a message to the topic
            client.publish(topic, "sd_unlock") #########################################################
        else:
            continue


def sd_thread():
    global sd_unlock, vit_gpt2_unlock, queue_prompt, broker_hostname, broker_port, topic, vit_caption, max_ncycles_prompt

    # Set up the client
    client = mqtt.Client()
    # Connect to the broker
    client.connect(broker_hostname, broker_port)

    ngen_prompt = max_ncycles_prompt+1
    sd_prompt = ""
    while 1:
        time.sleep(0.5)

        if sd_unlock and not vit_gpt2_unlock:
            if not len(queue_prompt) == 0 and ngen_prompt>max_ncycles_prompt:
                ngen_prompt = 0
                sd_prompt = queue_prompt[0]
                queue_prompt.pop(0) # remove this prompt from queue
            
            else:
                sd_prompt = vit_caption
                ngen_prompt+=1
                print("Number of generations: " + str(ngen_prompt))

            print("Generating image for prompt: " + sd_prompt)

            command = "python optimizedSD/optimized_txt2img.py --prompt " + "'" + sd_prompt + "'" + " --ddim_steps " + str(ddim_steps) + \
            " --outdir " + outdir + " --n_iter " + str(n_iter) + " --n_samples " + str(n_samples) + " --scale " + str(scale)
            os.system(command)
            print("Image Generated")

            # Publish a message to the topic
            client.publish(topic, "vit_gpt2_unlock")

            sd_unlock = False



def vit_gpt2_thread():
    global vit_gpt2_unlock, sd_unlock, broker_hostname, broker_port, topic, vit_caption

    # Set up the client
    client = mqtt.Client()
    # Connect to the broker
    client.connect(broker_hostname, broker_port)

    while 1:
        time.sleep(0.5)

        if vit_gpt2_unlock and not sd_unlock:
            print("----------------------------------------------")
            command = "python vit-gpt2-image-captioning/vit-gpt2-image-captioning.py --max_length " + str(max_length) + " --num_beams " + str(num_beams)
            os.system(command)

            # Open or create a file if exists and write the predicted caption
            with open("/output/ciclo_limite/caption.txt", "r") as text_file:
                vit_caption = text_file.readline()
            print("Caption for generated image:" + vit_caption)
            print("----------------------------------------------")

            # Publish a message to the topic
            client.publish(topic, "sd_unlock")

            vit_gpt2_unlock = False


# Define a callback function for when a message is received
def on_message(client, userdata, message):
    global sd_unlock, vit_gpt2_unlock

    print("Received message: ", str(message.payload.decode("utf-8")))

    if message.payload.decode("utf-8") == "sd_unlock":
        sd_unlock = True
        vit_gpt2_unlock = False
    elif message.payload.decode("utf-8") == "vit_gpt2_unlock":
        sd_unlock = False
        vit_gpt2_unlock = True
    else:
        sd_unlock = False
        vit_gpt2_unlock = False


if __name__ == "__main__":
    # Connect to the broker
    client.connect(broker_hostname, broker_port)
    # Subscribe to a topic
    client.subscribe(topic)
    # Set the callback function
    client.on_message = on_message
    # Start the client loop
    client.loop_start()



    thread_queue = Thread(target = queue_thread, args=[])
    thread_queue.start()

    thread_sd = Thread(target = sd_thread, args=[])
    thread_sd.start()

    thread_vit_gpt2 = Thread(target = vit_gpt2_thread, args=[])
    thread_vit_gpt2.start()

    while 1:
        time.sleep(5000)

