from torch import autocast
import torch
from diffusers import StableDiffusionPipeline
from diffusers import LMSDiscreteScheduler, EulerDiscreteScheduler
from diffusers import DDIMScheduler, DDPMScheduler, PNDMScheduler

def text2img_sdm(prompt, negative_prompt=None, device="cuda"):
    # scheduler = LMSDiscreteScheduler(
    #     beta_start=0.00085, 
    #     beta_end=0.012, 
    #     beta_schedule="scaled_linear"
    # )
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     "./third-parties/stable-diffusion-v1-4", 
    #     # revision="fp16", 
    #     # scheduler=scheduler,
    #     # torch_dtype=torch.float16,
    #     # use_auth_token=True
    # )

    model_id = "weights/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)

    pipe = pipe.to(device)

    with autocast("cuda"):
        image = pipe(prompt, negative_prompt=negative_prompt).images[0]  
    
    return image


if __name__=='__main__':
    prompt = "An astronaut riding a horse"    #"a kungfu panda sitting on a bench" #"a cozy living room"
    negative_prompt = 'blurry, bad art, blurred, text, watermark'
    image = text2img_sdm(prompt, negative_prompt=negative_prompt, device="cuda")
    image.save("test_"+prompt.replace(" ", "_")+".png")
