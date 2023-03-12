import urllib.request
import re
import time
import os
from threading import Thread

ddim_steps = 15
scale = 7.5
n_iter = 1
n_samples =  1
outdir = "outputs/txt2img-samples"

link = "https://rosmolarr.pythonanywhere.com/last"


queue_prompt = []
prev_prompt = ""
def queue_thread():
    global prev_prompt

    while 1:
        time.sleep(0.5)

        f = urllib.request.urlopen(link)
        txt = f.read().decode("utf-8")
        new_prompt = re.findall("<p>(.*?)<\/p>",txt)[0]

        if not new_prompt == prev_prompt:
            queue_prompt.append(new_prompt)
            print("New prompt: " + str(queue_prompt))
            prev_prompt = new_prompt
        else:
            continue

thread = Thread(target = queue_thread, args=[])
thread.start()

while 1:
    time.sleep(0.5)

    if queue_prompt:
        prompt = queue_prompt[0]

        print("Generating image for prompt: " + prompt)

        command = "python optimizedSD/optimized_txt2img.py --prompt " + "'" + prompt + "'" + " --ddim_steps " + str(ddim_steps) + \
        " --outdir " + outdir + " --n_iter " + str(n_iter) + " --n_samples " + str(n_samples) + " --scale " + str(scale)

        os.system(command)
        queue_prompt.pop(0)

        print("Image Generated")

        print("----------------------------------------------")
        print("Caption for generated image:")
        command = "python vit-gpt2-image-captioning/vit-gpt2-image-captioning.py"
        os.system(command)
        print("----------------------------------------------")

    else:
        continue