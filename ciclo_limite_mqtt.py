import urllib.request
import re
import time
import os
import paho.mqtt.client as mqtt
from deep_translator import GoogleTranslator
import random

import sys
sys.path.insert(1, '/optimizedSD')
from optimized_txt2img_class import stable_diffusion_model
sys.path.insert(1, '/vit-gpt2-image-captioning')
from vit_gpt2_image_captioning_class import vit_gpt2_image_captioning_model

########### Ciclo Limite Parameters ###########
max_ncycles_prompt = 0 # max number of cycles per prompt
link = "https://rosmolarr.pythonanywhere.com/last"
keywords = ["realistic","abstract",""]

########### Stable Diffusion Parameters ###########
ddim_steps = 5 # number of ddim sampling steps
scale = 4 # cfg scale can be seen as the “Creativity vs. Prompt” scale. Lower numbers give the AI more freedom to be creative, while higher numbers force it to stick more to the prompt (7.5 gives the best balance between creativity and generating what you want)
n_iter = 1 # sample this often
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
prev_prompt = ""
ngen_prompt = max_ncycles_prompt+1
sd_prompt = ""
###############################################


def queue_thread():
    global queue_prompt,prev_prompt,client
    
    try:
        f = urllib.request.urlopen(link)
        txt = f.read().decode("utf-8")
        new_prompt = re.findall("<p>(.*?)<\/p>",txt)[0]
        print("Prompt read from web: " + str(new_prompt))
    except:
        print("Web page not available")
        return

    if not new_prompt == prev_prompt:
        queue_prompt.append(new_prompt.strip().lower())
        print("New prompt: " + str(queue_prompt))
        prev_prompt = new_prompt

        sd_unlock = True
        # Publish a message to the topic
        client.publish(topic, "sd_unlock")
    else:
        # continue
        pass


def sd_thread(model):
    global sd_unlock, queue_prompt, ngen_prompt, \
            vit_caption, max_ncycles_prompt, sd_prompt, client

    if sd_unlock:
        if not len(queue_prompt) == 0 and ngen_prompt>=max_ncycles_prompt:
            ngen_prompt = 0
            sd_prompt = queue_prompt[0]
            queue_prompt.pop(0) # remove this prompt from queue

            client.publish(topic, sd_prompt)
        
        else:
            sd_prompt = vit_caption
            ngen_prompt+=1
            print("Number of generations: " + str(ngen_prompt))

        # Translate prompt
        sd_prompt = GoogleTranslator(source='auto', target='en').translate(sd_prompt)
        # Add random keyword
        sd_prompt = random.choice(keywords) + " " + sd_prompt

        print("Generating image for prompt: " + sd_prompt)

        # Generate image
        model.prompt = sd_prompt
        model.predict()
        print("Image Generated")

        sd_unlock = False
        # Publish a message to the topic
        client.publish(topic, "vit_gpt2_unlock")




def vit_gpt2_thread(model):
    global sd_unlock, vit_caption, client

    if not sd_unlock:
        print("----------------------------------------------")
        model.predict_caption()

        # Open a file if exists and read the predicted caption
        with open("/output/ciclo_limite/caption.txt", "r") as text_file:
            vit_caption = text_file.readline()
        print("Caption for generated image:" + vit_caption)
        print("----------------------------------------------")

        sd_unlock = True
        # Publish a message to the topic
        client.publish(topic, "sd_unlock")


if __name__ == "__main__":
    # Connect to the broker
    client.connect(broker_hostname, broker_port)
    # Subscribe to a topic
    client.subscribe(topic)
    # Start the client loop
    client.loop_start()

    sd_model = stable_diffusion_model(ddim_steps=ddim_steps, outdir=outdir, n_iter=n_iter, n_samples=n_samples, scale=scale)
    vit_gpt_model = vit_gpt2_image_captioning_model(max_length=max_length,num_beams=num_beams)

    while 1:
        time.sleep(0.1)

        queue_thread()
        sd_thread(sd_model)
        time.sleep(10)
        vit_gpt2_thread(vit_gpt_model) 