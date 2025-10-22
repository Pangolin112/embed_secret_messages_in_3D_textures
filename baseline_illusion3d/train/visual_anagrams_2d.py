import mediapy as mp

import os

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from diffusers import DiffusionPipeline

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_stage_1, sample_stage_2
from visual_anagrams.animate import animate_two_view


def im_to_np(im):
  im = (im / 2 + 0.5).clamp(0, 1)
  im = im.detach().cpu().permute(1, 2, 0).numpy()
  im = (im * 255).round().astype("uint8")
  return im


def visual_anagrams_2d():
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
    output_path = './outputs/visual_anagrams_2d'
    os.makedirs(output_path, exist_ok=True)
    
    # stage 1
    stage_1 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-M-v1.0",
        variant="fp16",
        torch_dtype=torch.float16,
    )
    stage_1.enable_model_cpu_offload()
    stage_1 = stage_1.to(device)

    # stage 2
    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-M-v1.0",
        text_encoder=None,
        variant="fp16",
        torch_dtype=torch.float16,
    )
    stage_2.enable_model_cpu_offload()
    stage_2 = stage_2.to(device)

    # stage 3
    stage_3 = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                torch_dtype=torch.float16
            )
    stage_3.enable_model_cpu_offload()
    stage_3 = stage_3.to(device)

    # views
    views = get_views(['identity', 'rotate_180'])
    # views = get_views(['identity', 'rotate_cw'])
    # views = get_views(['identity', 'rotate_ccw'])
    # views = get_views(['identity', 'flip'])
    #views = get_views(['identity', 'negate'])
    # views = get_views(['identity', 'skew'])
    #views = get_views(['identity', 'patch_permute'])
    # views = get_views(['identity', 'pixel_permute'])
    # views = get_views(['identity', 'inner_circle'])
    # views = get_views(['identity', 'square_hinge'])
    # views = get_views(['identity', 'jigsaw'])

    # prompts
    # prompt_1 = 'painting of a snowy mountain village'
    # prompt_2 = 'painting of a horse'

    # prompt_1 = 'an oil painting of an old man'
    # prompt_2 = 'an oil painting of people at a campfire'

    prompt_1 = 'the word "happy", cursive writing'
    prompt_2 = 'the word "holiday", cursive writing'

    # prompt_1 = 'painting of a cow'
    # prompt_2 = 'painting of a dog'

    # Embed prompts using the T5 model
    prompts = [prompt_1, prompt_2]
    prompt_embeds = [stage_1.encode_prompt(prompt) for prompt in prompts]
    prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
    prompt_embeds = torch.cat(prompt_embeds)
    negative_prompt_embeds = torch.cat(negative_prompt_embeds)  # These are just null embeds

    # sample stage 1
    image_64 = sample_stage_1(
        stage_1,
        prompt_embeds,
        negative_prompt_embeds,
        views,
        num_inference_steps=30,
        guidance_scale=10.0,
        reduction='mean',
        generator=generator
    )
    mp.show_images([im_to_np(view.view(image_64[0])) for view in views])

    # sample stage 2
    image_256 = sample_stage_2(
        stage_2,
        image_64,
        prompt_embeds,
        negative_prompt_embeds,
        views,
        num_inference_steps=30,
        guidance_scale=10.0,
        reduction='mean',
        noise_level=50,
        generator=generator
    )
    mp.show_images([im_to_np(view.view(image_256[0])) for view in views])

    # sample stage 2
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

    # Limit display size, otherwise it's too large for most screens
    # mp.show_images([im_to_np(view.view(image_1024[0])) for view in views], width=400)

    # make video, no use for now
    # # animate the illusion
    # #image = image_64
    # #image = image_256  
    # image = image_1024

    # # Get size
    # im_size = image.shape[-1]
    # frame_size = int(im_size * 1.5)

    # # Make save path
    # save_video_path = output_path + '/animation_visual_anagrams.mp4'

    # # Convert to PIL
    # pil_image = TF.to_pil_image(image[0] / 2. + 0.5)

    # # Make the animation
    # animate_two_view(
    #     pil_image,
    #     views[1], # Use the non-identity view to transform
    #     prompt_1,
    #     prompt_2,
    #     save_video_path=save_video_path,
    #     hold_duration=120,
    #     text_fade_duration=10,
    #     transition_duration=45,
    #     im_size=im_size,
    #     frame_size=frame_size,
    # )

    # # Display the video (using max width of 600 so will fit on most screens)
    # mp.show_video(mp.read_video(save_video_path), fps=30, width=min(600, frame_size))