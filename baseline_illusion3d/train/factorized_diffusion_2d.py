import gc
import mediapy as mp

import os

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from diffusers import DiffusionPipeline

from transformers import T5EncoderModel

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_stage_1, sample_stage_2
from visual_anagrams.utils import add_args, save_illusion, save_metadata
from visual_anagrams.animate import animate_two_view


def im_to_np(im):
  im = (im / 2 + 0.5).clamp(0, 1)
  im = im.detach().cpu().permute(1, 2, 0).numpy()
  im = (im * 255).round().astype("uint8")
  return im


# Garbage collection function to free memory
def flush():
    gc.collect()
    torch.cuda.empty_cache()


def factorized_diffusion_2d():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    seed = 99
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # Set paths
    output_path = './outputs/factorized_diffusion_2d'
    os.makedirs(output_path, exist_ok=True)

    text_encoder = T5EncoderModel.from_pretrained(
        "DeepFloyd/IF-I-L-v1.0",
        subfolder="text_encoder",
        device_map="auto",
        variant="fp16",
        torch_dtype=torch.float16,
    )

    pipe = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-L-v1.0",
        text_encoder=text_encoder,  # pass the previously instantiated text encoder
        unet=None                   # do not use a UNet here, as it uses too much memory
    )
    pipe = pipe.to(device)

    # prompts
    # prompt_2 = 'a painting of a panda'
    # prompt_1 = 'a painting of a flower arrangement'

    # prompt_2 = 'a photo of houseplants'
    # prompt_1 = 'a photo of marilyn monroe'

    prompt_2 = 'a photo of rome'
    prompt_1 = 'a photo of a yin yang'

    prompts = [prompt_1, prompt_2]
    prompt_embeds = [pipe.encode_prompt(prompt) for prompt in prompts]
    prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
    prompt_embeds = torch.cat(prompt_embeds)
    negative_prompt_embeds = torch.cat(negative_prompt_embeds)  # These are just null embeds

    del text_encoder
    del pipe
    flush()
    flush()   # For some reason we need to do this twice

    # Load DeepFloyd IF stage I
    stage_1 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-L-v1.0",
        text_encoder=None,
        variant="fp16",
        torch_dtype=torch.float16,
    )
    stage_1.enable_model_cpu_offload()
    stage_1.to(device)

    # Load DeepFloyd IF stage II
    stage_2 = DiffusionPipeline.from_pretrained(
                    "DeepFloyd/IF-II-L-v1.0",
                    text_encoder=None,
                    variant="fp16",
                    torch_dtype=torch.float16,
                )
    stage_2.enable_model_cpu_offload()
    stage_2.to(device)

    # Load DeepFloyd IF stage III
    # (which is just Stable Diffusion 4x Upscaler)
    stage_3 = DiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-x4-upscaler",
                    torch_dtype=torch.float16
                )
    stage_3.enable_model_cpu_offload()

    views = get_views(['low_pass', 'high_pass'])   # Hybrids
    #views = get_views(['motion', 'motion_res'])    # Motion Hybrids
    #views = get_views(['grayscale', 'color'])      # Color Hybrids

    image_64 = sample_stage_1(
        stage_1,
        prompt_embeds,      # Replace with different prompts
        negative_prompt_embeds,
        views,
        num_inference_steps=30,
        guidance_scale=10.0,
        reduction='sum',
        generator=generator
    )

    # Show result
    mp.show_images([im_to_np(view.save_view(image_64[0]))
                    if hasattr(view, 'save_view')
                    else im_to_np(view.view(image_64[0]))
                    for view in views])
    
    image_256 = sample_stage_2(
        stage_2,
        image_64,
        prompt_embeds,      # Replace with different prompts
        negative_prompt_embeds,
        views,
        num_inference_steps=30,
        guidance_scale=10.0,
        reduction='sum',
        noise_level=50,
        generator=generator
    )

    # Show result
    mp.show_images([im_to_np(view.view(image_256[0])) for view in views])

    image_1024 = stage_3(
        prompt=prompts[0],  # Note this is a string, and not an embedding
        image=image_256,
        noise_level=0,
        output_type='pt',
        generator=generator
    ).images
    image_1024 = image_1024 * 2 - 1

    save_image((image_1024[0] / 2 + 0.5).clamp(0, 1), output_path + f'/image_{prompt_1}.png')
    save_image((views[1].view(image_1024[0]) / 2 + 0.5).clamp(0, 1), output_path + f'/image_{prompt_2}.png')

    # # Limit display size, otherwise it's too large for most screens
    # mp.show_images([im_to_np(view.view(image_1024[0])) for view in views], width=400)

    # #image = image_64
    # #image = image_256
    # image = image_1024

    # # Get size
    # im_size = image.shape[-1]
    # frame_size = int(im_size * 1.5)

    # # Make save path
    # save_video_path = './outputs/animation_Factorized_Diffusion.mp4'

    # # Convert to PIL
    # pil_image = TF.to_pil_image(image[0] / 2. + 0.5)

    # # Make the animation
    # animate_two_view(
    #             pil_image,
    #             views[1], # Either view should work
    #             prompt_2, # NOTE: Prompts may need to be switched
    #             prompt_1,
    #             save_video_path=save_video_path,
    #             hold_duration=120,
    #             text_fade_duration=10,
    #             transition_duration=45,
    #             im_size=im_size,
    #             frame_size=frame_size,
    #         )

    # # Display the video (using max width of 600 so will fit on most screens)
    # mp.show_video(mp.read_video(save_video_path), fps=30, width=min(600, frame_size))



    