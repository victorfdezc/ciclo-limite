'''
This is the simplest Ciclo Limite script. No MQTT. No threading.
'''

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
min_sec = 10 # minimum number of seconds showing the image or text (must be different than zero)
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
text_path = "/output/ciclo_limite/caption.txt"

############# Globals Definition #############
queue_prompt = []
vit_caption = ""
client = mqtt.Client() # Set up the MQTT client
prev_prompt = ""
process_new_prompt = True
ngen_prompt = max_ncycles_prompt+1
sd_prompt = ""
###############################################


def web_prompt_queue():
    global queue_prompt, prev_prompt, process_new_prompt, client, ngen_prompt, max_ncycles_prompt
    
    process_new_prompt = False
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

        if ngen_prompt>=max_ncycles_prompt:
            process_new_prompt = True
    else:
        return


def stable_diffusion(model):
    global queue_prompt, process_new_prompt, sd_prompt, vit_caption, ngen_prompt, max_ncycles_prompt

    if process_new_prompt:
        # If there is a new prompt and the number of iterations of this cycle has
        # reached its maximum, begin a new cycle with new promt
        ngen_prompt = 0
        sd_prompt = queue_prompt[0]
        queue_prompt.pop(0) # remove this prompt from queue
    
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



def vit_gpt2(model):
    global vit_caption, process_new_prompt

    # First check if there is any new promt, so we display it on screen
    web_prompt_queue()

    print("----------------------------------------------")
    if not process_new_prompt:
        model.predict_caption()

        # Open a file if exists and read the predicted caption
        with open(text_path, "r") as text_file:
            vit_caption = text_file.readline()
        print("Caption for generated image:" + vit_caption)

    else:
        # Write new prompt to text file
        with open(text_path, "w+") as text_file:
            print("Detected new prompt to process... aborting git-gpt2 execution")
            text_file.write(queue_prompt[0])
    print("----------------------------------------------")


if __name__ == "__main__":
    # Create models
    sd_model = stable_diffusion_model(ddim_steps=ddim_steps, outdir=outdir, n_iter=n_iter, n_samples=n_samples, scale=scale)
    vit_gpt_model = vit_gpt2_image_captioning_model(max_length=max_length,num_beams=num_beams)

    while 1:
        time.sleep(min_sec) # TODO: compute time elapsed
        vit_gpt2(vit_gpt_model)
        #time.sleep(min_sec) # TODO: compute time elapsed
        stable_diffusion(sd_model)