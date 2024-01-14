from io import BytesIO

import numpy as np
import PIL, cv2, imageio
import requests
import torch
# from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from torch import autocast, unsqueeze


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

def text_inpaint_sdm(init_image, mask_image, prompt, device="cuda", strength=0.75, guidance_scale=7.5, num_images_per_prompt=1):
    # model_id_or_path = "./third-parties/stable-diffusion-v1-4"
    model_id_or_path = "weights/stable-diffusion-2-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id_or_path,
        # revision="fp16", 
        torch_dtype=torch.float16,
        # use_auth_token=True
    )
    pipe = pipe.to(device)

    with autocast("cuda"):
        images = pipe(prompt=prompt, image=init_image, mask_image=mask_image, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt).images #0.75
    
    return images[0]

def text_inpaint_sdm2(init_image, mask_image, prompt, negative_prompt=None, device="cuda", strength=0.75, guidance_scale=7.5, num_images_per_prompt=1):
    model_id_or_path = "weights/stable-diffusion-2-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)

    with autocast("cuda"):
        images = pipe(prompt=prompt, image=init_image, mask_image=mask_image, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt).images 
    
    return images

# to define a class which is same as 'text_inpaint_sdm2'
class text2inpainting_sdm():
    def __init__(self, device="cuda"):
        super().__init__()
        model_id_or_path = "weights/stable-diffusion-2-inpainting"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        model_id_or_path,
                        torch_dtype=torch.float16,
                    )
        self.pipe = self.pipe.to(device)
        self.device = device

    def get_text_embeds(self, prompt):
        text_input = self.pipe.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=self.pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings
        
    def sdm(self, init_image, mask_image, prompt, negative_prompt=None, guidance_scale=7.5, num_images_per_prompt=1):
        with autocast("cuda"):
            images = self.pipe(prompt=prompt, image=init_image, mask_image=mask_image, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt).images 
        return images

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    # mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    # init_image = download_image(img_url).resize((512, 512))
    # mask_image = download_image(mask_url).resize((512, 512))

    init_image = PIL.Image.open('/apdcephfs/share_1330077/eckertzhang/Dataset/data_for_text2nerf/000_text_a_beautiful_garden_with_a_fountain/DIBR_gt/warped/00002.png').convert("RGB").resize((512, 512))
    mask_image = PIL.Image.open('/apdcephfs/share_1330077/eckertzhang/Dataset/data_for_text2nerf/000_text_a_beautiful_garden_with_a_fountain/DIBR_gt/mask_inv/00002.png').convert("RGB").resize((512, 512))
    # mask = np.array(mask_image)[:,:,0]
    # mask[mask>0] = 255
    # mask_image = PIL.Image.fromarray(mask)

    # init_image = cv2.imread('ziyu0510/0002_in_img.jpg', cv2.IMREAD_UNCHANGED)
    # mask_image = cv2.imread('ziyu0510/0002_in_mask.jpg', cv2.IMREAD_UNCHANGED)
    # init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB)
    # init_image = cv2.resize(init_image, (512,512), interpolation=cv2.INTER_LINEAR)
    # mask_image = cv2.resize(mask_image, (512,512), interpolation=cv2.INTER_NEAREST)
    # mask_image[mask_image>100]=255
    # mask_image[mask_image<=100]=0
    # img = init_image.copy()
    # for i in range(3):
    #     img[:,:,i] = img[:,:,i]*(1-mask_image/255) + mask_image
    # imageio.imwrite('masked_img.png', img)
    # init_image = PIL.Image.fromarray(img)
    # mask_image = PIL.Image.fromarray(mask_image)

    # mask_single = (np.array(mask_image.convert('L'))/255).astype(np.uint8)
    # init_image = np.array(init_image).transpose([2,0,1])*(1-mask_single)
    # # init_image = init_image + (255*mask_single).astype(np.uint8)[np.newaxis,:,:].repeat(3, axis=0)
    # init_image = PIL.Image.fromarray(init_image.transpose([1,2,0]))
    # mask_image = PIL.Image.fromarray(np.array(mask_image)[:,:,0])
    # init_image.save('results/00_test_img/test_initial_under_black_.png')

    prompt = "a beautiful garden with a fountain"    #"a cat sitting on a bench" #"a cozy living room"
    negative_prompt = 'blurry, bad art, blurred, text, watermark'
    guidance_scale=7.5
    num = 5
    images = text_inpaint_sdm2(init_image, mask_image, prompt, negative_prompt=negative_prompt, device="cuda", guidance_scale=guidance_scale, num_images_per_prompt=num)
    for i in range(len(images)):
        images[i].save(f"00test_sdm2_inpainting_{guidance_scale}_{i}.png")