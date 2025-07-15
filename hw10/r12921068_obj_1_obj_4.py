import json
import os

import torch.nn.functional as F
from accelerate.utils import set_seed

from diffusers.pipelines import BlipDiffusionPipeline
from diffusers.utils import load_image

set_seed(42)

def InferenceBlipDiffusion(
    pipe,
    cond_image_path,
    name,
    text_prompt_input,
    guidance_scale,
    num_inference_steps,
    saveDir,
    num_images
):
    """
    Performs inference using a Blip Diffusion pipeline to generate images based on a conditional image and text prompt.

    Args:
        pipe: The Blip Diffusion pipeline object.
        cond_image_path (str): The file path to the conditioning image. This image guides the style and content of the generated images.
        name (str): A descriptive name associated with the subject or concept in the conditioning image. This name is used for both the conditioning and target subjects.
        text_prompt_input (str): The text prompt that provides additional information and guides the generation of the new images.
        guidance_scale (float): A value controlling the influence of the text prompt on the generated images. Higher values enforce the prompt more strongly.
        num_inference_steps (int): The number of denoising steps to perform during the diffusion process. More steps generally lead to higher quality images but take longer to generate.
        saveDir (str): The directory where the generated images will be saved. The function will create this directory if it doesn't exist.
        num_images (int): The number of images to generate.

    Returns:
        None. The generated images are saved to the specified `saveDir`.
    """

    os.makedirs(saveDir, exist_ok = True) # create output directory
    # prepare arguments for BLIP Diffusion
    cond_subject = name
    tgt_subject = name
    cond_image = load_image(cond_image_path)
    negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

    for i in range(num_images):
        output = pipe(
            text_prompt_input,
            cond_image,
            cond_subject,
            tgt_subject,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            neg_prompt=negative_prompt,
            height=512,
            width=512,
        ).images
        output[0].save(f"{saveDir}/{i}.jpg")

def main():
    blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained("Salesforce/blipdiffusion", mean=None, std=None).to("cuda")

    with open("ml2025-hw10/metadata.json", "r") as f:
        objects = json.load(f)

    ##################### TODO: Tune hyperparameters here ##########################

    num_inference_steps = 100   # The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
    guidance_scale = 4.0        # Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.

    ################################################################################

    num_images = 15             # The number of images you want to generate.
                                # WARNING: You MUST to keep it 15 if you want to generate the images for submission.
                                # But you can reduce it when tuning hyperparameters to speed up the process
    output_dir = "results"

    # iterate through each of the 6 objects to customize
    for (obj, info) in objects.items():
        # If you only want to generate results for specific object remove the "#" below and adjust the list
        #if (obj not in ["object-1", "object-2", "object-3", "object-4", "object-5", "object-6"]): continue
        if (obj in ["object-1", "object-2", "object-3", "object-4"]):
            InferenceBlipDiffusion(
                pipe = blip_diffusion_pipe,
                cond_image_path = info["path"],
                name = info["name"],
                text_prompt_input = info["text_cond"],
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps,
                saveDir = os.path.join(output_dir, obj),
                num_images = num_images
            )

if __name__ == "__main__":
    main()
