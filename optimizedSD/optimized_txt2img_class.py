import argparse, os, re
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging
# from samplers import CompVisDenoiser
logging.set_verbosity_error()


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

class stable_diffusion_model:
    def __init__(self,prompt="a painting of a virus monster playing guitar",outdir="outputs/txt2img-samples",skip_grid=False,\
                skip_save=False,ddim_steps=50,fixed_code=False,ddim_eta=0.0,n_iter=1,H=512,W=512,C=4,f=8,n_samples=5,n_rows=0,scale=7.5,device="cuda",\
                seed=None,unet_bs=1,turbo=False,precision="autocast",format="png",sampler="plms",DEFAULT_CKPT = "models/ldm/stable-diffusion-v1/model.ckpt",\
                from_file=False):

        self.config = "optimizedSD/v1-inference.yaml"

        self.prompt = prompt
        self.outdir = outdir
        self.skip_grid = skip_grid
        self.skip_save = skip_save
        self.ddim_steps = ddim_steps
        self.fixed_code = fixed_code
        self.ddim_eta = ddim_eta
        self.n_iter = n_iter
        self.H = H
        self.W = W
        self.C = C
        self.f = f
        self.n_samples = n_samples
        self.n_rows = n_rows
        self.scale = scale
        self.device = device
        self.seed = seed
        self.unet_bs = unet_bs
        self.turbo = turbo
        self.precision = precision
        self.format = format
        self.sampler = sampler
        self.ckpt = DEFAULT_CKPT
        self.from_file = from_file

        # Load models and create necessary variables
        self.sd = load_model_from_config(f"{self.ckpt}")
        self.config = OmegaConf.load(f"{self.config}")
        os.makedirs(self.outdir, exist_ok=True)
        self.outpath = self.outdir

        if self.seed == None:
            self.seed = randint(0, 1000000)
        seed_everything(self.seed)

        # # Logging
        # logger(vars(opt), log_csv = "logs/txt2img_logs.csv")


    def predict(self):
        tic = time.time()

        li, lo = [], []
        for key, value in self.sd.items():
            sp = key.split(".")
            if (sp[0]) == "model":
                if "input_blocks" in sp:
                    li.append(key)
                elif "middle_block" in sp:
                    li.append(key)
                elif "time_embed" in sp:
                    li.append(key)
                else:
                    lo.append(key)
        for key in li:
            self.sd["model1." + key[6:]] = self.sd.pop(key)
        for key in lo:
            self.sd["model2." + key[6:]] = self.sd.pop(key)

        model = instantiate_from_config(self.config.modelUNet)
        _, _ = model.load_state_dict(self.sd, strict=False)
        model.eval()
        model.unet_bs = self.unet_bs
        model.cdevice = self.device
        model.turbo = self.turbo

        modelCS = instantiate_from_config(self.config.modelCondStage)
        _, _ = modelCS.load_state_dict(self.sd, strict=False)
        modelCS.eval()
        modelCS.cond_stage_model.device = self.device

        modelFS = instantiate_from_config(self.config.modelFirstStage)
        _, _ = modelFS.load_state_dict(self.sd, strict=False)
        modelFS.eval()

        if self.device != "cpu" and self.precision == "autocast":
            model.half()
            modelCS.half()

        start_code = None
        if self.fixed_code:
            start_code = torch.randn([self.n_samples, self.C, self.H // self.f, self.W // self.f], device=self.device)


        batch_size = self.n_samples
        n_rows = self.n_rows if self.n_rows > 0 else batch_size
        if not self.from_file:
            assert self.prompt is not None
            prompt = self.prompt
            print(f"Using prompt: {prompt}")
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {self.from_file}")
            with open(self.from_file, "r") as f:
                text = f.read()
                print(f"Using prompt: {text.strip()}")
                data = text.splitlines()
                data = batch_size * list(data)
                data = list(chunk(sorted(data), batch_size))


        if self.precision == "autocast" and self.device != "cpu":
            precision_scope = autocast
        else:
            precision_scope = nullcontext

        seeds = ""
        with torch.no_grad():

            all_samples = list()
            for n in trange(self.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):

                    # sample_path = os.path.join(self.outpath, "_".join(re.split(":| ", prompts[0])))[:150]
                    sample_path = os.path.join(self.outpath, "ciclo_limite")
                    os.makedirs(sample_path, exist_ok=True)
                    # base_count = len(os.listdir(sample_path))
                    base_count = 0

                    with precision_scope("cuda"):
                        modelCS.to(self.device)
                        uc = None
                        if self.scale != 1.0:
                            uc = modelCS.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        subprompts, weights = split_weighted_subprompts(prompts[0])
                        if len(subprompts) > 1:
                            c = torch.zeros_like(uc)
                            totalWeight = sum(weights)
                            # normalize each "sub prompt" and add it
                            for i in range(len(subprompts)):
                                weight = weights[i]
                                # if not skip_normalize:
                                weight = weight / totalWeight
                                c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                        else:
                            c = modelCS.get_learned_conditioning(prompts)

                        shape = [self.n_samples, self.C, self.H // self.f, self.W // self.f]

                        if self.device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            modelCS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)

                        samples_ddim = model.sample(
                            S=self.ddim_steps,
                            conditioning=c,
                            seed=self.seed,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=self.scale,
                            unconditional_conditioning=uc,
                            eta=self.ddim_eta,
                            x_T=start_code,
                            sampler = self.sampler,
                        )

                        modelFS.to(self.device)

                        print(samples_ddim.shape)
                        print("saving images")
                        for i in range(batch_size):

                            x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                            x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                            # Image.fromarray(x_sample.astype(np.uint8)).save(
                            #     os.path.join(sample_path, "seed_" + str(self.seed) + "_" + f"{base_count:05}.{self.format}")
                            # )
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, "image_" + f"{base_count:05}.{self.format}")
                            )
                            seeds += str(self.seed) + ","
                            self.seed += 1
                            base_count += 1

                        if self.device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            modelFS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)
                        del samples_ddim
                        print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

        toc = time.time()

        time_taken = (toc - tic) / 60.0

        print(
            (
                "Samples finished in {0:.2f} minutes and exported to "
                + sample_path
                + "\n Seeds used = "
                + seeds[:-1]
            ).format(time_taken)
        )
