import mediapy as mp

import os

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from diffusers import DiffusionPipeline

from visual_anagrams_sd.views import get_views
from visual_anagrams_sd.samplers import sample_sd
from visual_anagrams_sd.animate import animate_two_view


def factorized_diffusion_2d_sd():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    seed = 99
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # num_inference_steps = 30
    num_inference_steps = 100

    # Set paths
    output_path = './outputs/factorized_diffusion_2d_sd'
    os.makedirs(output_path, exist_ok=True)

    # parameters
    dtype = torch.float32
    dtype_half = torch.float16

    # sd 1.5
    model_path = "runwayml/stable-diffusion-v1-5"
    diffusion_model = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype_half, low_cpu_mem_usage=True).to(device)
    diffusion_model.enable_model_cpu_offload()

    # views
    views = get_views(['low_pass', 'high_pass'])   # Hybrids
    
    # prompts
    # prompt_1 = 'a painting of a panda'
    # prompt_2 = 'a painting of a flower arrangement'

    # prompt_2 = 'a photo of houseplants'
    # prompt_1 = 'a photo of marilyn monroe'

    prompt_2 = 'a photo of rome'
    prompt_1 = 'a photo of a yin yang'

    image = sample_sd(
        diffusion_model,
        prompt_1,
        prompt_2,
        views,
        num_inference_steps=num_inference_steps,
        guidance_scale=10.0,
        reduction='sum',
        generator=generator
    )

    save_image((image / 2 + 0.5).clamp(0, 1), output_path + f'/image_{prompt_1}_num_inference_steps_{num_inference_steps}.png')
    save_image((views[1].view(image) / 2 + 0.5).clamp(0, 1), output_path + f'/image_{prompt_2}_num_inference_steps_{num_inference_steps}.png')

    # # make video, no use for now
    # # Get size
    # im_size = image.shape[-1]
    # frame_size = int(im_size * 1.5)

    # # Make save path
    # save_video_path = output_path + '/animation_visual_anagrams_sd.mp4'

    # # Convert to PIL
    # pil_image = TF.to_pil_image(image[0] / 2. + 0.5)

    # # Make the animation
    # animate_two_view(
    #     pil_image,
    #     views[1], # Use the non-identity view to transform
    #     prompt_2,
    #     prompt_1,
    #     save_video_path=save_video_path,
    #     hold_duration=120,
    #     text_fade_duration=10,
    #     transition_duration=45,
    #     im_size=im_size,
    #     frame_size=frame_size,
    # )

    # # Display the video (using max width of 600 so will fit on most screens)
    # mp.show_video(mp.read_video(save_video_path), fps=30, width=min(600, frame_size))