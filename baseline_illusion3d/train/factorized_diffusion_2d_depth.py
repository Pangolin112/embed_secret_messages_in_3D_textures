import mediapy as mp

import os

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler
from diffusers.utils import load_image

from visual_anagrams_sd.views import get_views
from visual_anagrams_sd.samplers import sample_sd, sample_sd_depth, sample_sd_depth_triple
from visual_anagrams_sd.animate import animate_two_view

from utils.pytorch3d_uv_utils import (
    generate_depth_tensor_from_mesh,
    generate_uv_mapping_mesh,
    import_smart_uv_mesh
)


def factorized_diffusion_2d_depth():
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

    # data directory
    DATA_DIR = "./data"
    obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow_rotated.obj")
    ply_filename = os.path.join(DATA_DIR, "ScanNetpp/meshes/49a82360aa/mesh_uv.ply")

    # Set paths
    output_path = './outputs/factorized_diffusion_2d_depth'
    os.makedirs(output_path, exist_ok=True)

    # parameters
    dtype = torch.float32
    dtype_half = torch.float16

    # sd 1.5
    controlnet_name = "lllyasviel/control_v11f1p_sd15_depth"
    controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=dtype_half, low_cpu_mem_usage=True).to(device)
    model_path = "runwayml/stable-diffusion-v1-5"
    diffusion_model = StableDiffusionControlNetPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=dtype_half, low_cpu_mem_usage=True).to(device)
    diffusion_model.enable_model_cpu_offload()
    
    controlnet = diffusion_model.controlnet.to(dtype_half)
    controlnet.requires_grad_(False)
    controlnet.enable_gradient_checkpointing()

    # get or generate uv parameterization
    latent_texture_size = 256
    latent_channels = 3
    num_views = 300
    render_size = 512
    faces_per_pixel = 1
    dist = 3.0

    conditioning_scale = 1.0

    num_views_eval = 16
    # new_mesh = generate_uv_mapping_mesh(obj_filename, output_path, device, latent_texture_size, latent_channels)
    # scene_scale = 10.0
    # new_mesh = import_smart_uv_mesh(ply_filename, output_path, device, latent_texture_size, latent_channels, scene_scale)

    # depth_tensor_list = generate_depth_tensor_from_mesh(dtype_half, num_views_eval, num_views, render_size, faces_per_pixel, dist, device, new_mesh)

    # depth_tensor = depth_tensor_list[0].to(device)

    depth_image = load_image('./data/depth_images/depth_scene.png')
    depth_image = depth_image.resize((render_size, render_size), resample=0)
    depth_tensor = TF.to_tensor(depth_image).unsqueeze(0).to(device).to(dtype_half)

    save_image((depth_tensor / 2 + 0.5).clamp(0, 1), output_path + f'/depth_tensor.png')

    # views
    views = get_views(['low_pass', 'high_pass'])   # Hybrids
    # views = get_views(['triple_low_pass', 'triple_medium_pass', 'triple_high_pass'])
    
    # prompts
    # prompt_1 = 'a painting of a panda'
    # prompt_2 = 'a painting of a flower arrangement'

    # prompt_1 = 'a photo of marilyn monroe'
    # prompt_2 = 'a photo of houseplants'
    
    # prompt_1 = 'a photo of a yin yang'
    # prompt_2 = 'a photo of rome'

    # prompt_1 = 'a photo of a cat'
    # prompt_2 = 'a photo of a cow'

    prompt_1 = 'a photo of a ancient ruins'
    prompt_2 = 'a photo of a japanese style living room'

    # prompt_1 = 'a photo of a castle'
    # prompt_2 = 'a photo of a scientific style living room'

    # prompt_1 = 'a photo of a panda'
    # prompt_2 = 'a photo of a dog'
    # prompt_3 = 'a photo of a japanese style living room'
    
    image = sample_sd_depth(
        diffusion_model,
        controlnet,
        prompt_1,
        prompt_2,
        depth_tensor,
        views,
        num_inference_steps=num_inference_steps,
        guidance_scale=10.0,
        conditioning_scale=conditioning_scale,
        reduction='sum',
        generator=generator
    )

    # image = sample_sd_depth_triple(
    #     diffusion_model,
    #     controlnet,
    #     prompt_1,
    #     prompt_2,
    #     prompt_3,
    #     depth_tensor,
    #     views,
    #     num_inference_steps=num_inference_steps,
    #     guidance_scale=10.0,
    #     conditioning_scale=conditioning_scale,
    #     reduction='sum',
    #     generator=generator
    # )

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