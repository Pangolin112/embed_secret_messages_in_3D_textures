from base64 import b64encode

import torch
from torchvision.utils import save_image
import os
        
from diffusers import DDIMScheduler, ControlNetModel as ControlNetModel, StableDiffusionControlNetPipeline as DiffusionPipeline
from huggingface_hub import notebook_login

import numpy as np
import cv2

from pathlib import Path
from torch import autocast
from tqdm.auto import tqdm
from transformers import logging

# utils
from utils.pytorch3d_uv_utils import (
    dataset_prepare,
    generate_uv_coords_from_mesh,
    generate_depth_tensor_from_mesh,
    plot_losses,
    generate_uv_mapping_mesh,
    query_texture,
    save_img_results,
    inference,
    hirarchical_inference
)


def IllusionDiffusionDepth():
    torch.manual_seed(1)
    if not (Path.home()/'.huggingface'/'token').exists(): notebook_login()

    # Supress some unnecessary warnings when loading the CLIPTextModel
    logging.set_verbosity_error()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load stable diffusion model
    dtype = torch.float32
    dtype_half = torch.float16
    model_path = "runwayml/stable-diffusion-v1-5"
    controlnet_name = "lllyasviel/control_v11f1p_sd15_depth"
    controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=dtype_half, low_cpu_mem_usage=True).to(device)
    diffusion_model = DiffusionPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=dtype_half, low_cpu_mem_usage=True).to(device)
    diffusion_model.enable_model_cpu_offload()
    controlnet = diffusion_model.controlnet.to(dtype_half)
    controlnet.requires_grad_(False)
    # components
    tokenizer = diffusion_model.tokenizer
    text_encoder = diffusion_model.text_encoder
    vae = diffusion_model.vae
    unet = diffusion_model.unet.to(dtype_half)
    # save VRAM
    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()
    # set device
    vae = vae.to(device).requires_grad_(False)
    text_encoder = text_encoder.to(device).requires_grad_(False)
    unet = unet.to(device).requires_grad_(False)
    # set scheduler
    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=dtype_half)
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    # set timesteps
    num_train_timesteps = len(scheduler.betas)
    scheduler.set_timesteps(num_train_timesteps)

    DATA_DIR = "./data"
    obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")
    # output directory
    output_path = './outputs/IllusionDiffusionDepth'
    os.makedirs(output_path, exist_ok=True)

    # get or generate uv parameterization
    latent_texture_size = 256
    latent_channels = 3
    num_views = 300
    render_size = 512
    faces_per_pixel = 1
    dist = 2.7

    conditioning_scale = 0.5

    num_views_eval = 16
    new_mesh = generate_uv_mapping_mesh(obj_filename, output_path, device, latent_texture_size, latent_channels)

    depth_tensor_list = generate_depth_tensor_from_mesh(num_views_eval, num_views, render_size, faces_per_pixel, dist, device, new_mesh)

    # Generate images
    first_prompt = "A bird" #@param {type:"string"}
    second_prompt = "A cow" #@param {type:"string"}
    rotation_deg = "90" #@param [90, 180]
    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    #@markdown More steps improves quality but takes longer.
    num_inference_steps = 500 #@param {type:"integer"}
    guidance_scale = 7.5 #@param {type:"slider", min:2, max:15, step:0.5}
    seed = 42 #@param {type:"integer"}
    generator = torch.manual_seed(seed)   
    batch_size = 1
    rotate = int(rotation_deg) // 90

    depth_tensor = depth_tensor_list[0].to(device)
    depth_tensor_rotated = torch.rot90(depth_tensor, rotate, [2, 3])

    save_image((depth_tensor / 2 + 0.5).clamp(0, 1), output_path + f'/depth_tensor.png')
    save_image((depth_tensor_rotated / 2 + 0.5).clamp(0, 1), output_path + f'/depth_tensor_rotated.png')

    # for seed in [0,1,2,3 ,4,5,6,7]:
    # 0,3,5    
    for seed in [0]:
        print(seed)
        generator = torch.manual_seed(seed)

        # Prep text 
        text_input = tokenizer([first_prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input_2 = tokenizer([second_prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
            test_embeddings_2 = text_encoder(text_input_2.input_ids.to(device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0] 
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_embeddings_2 = torch.cat([uncond_embeddings, test_embeddings_2])

        # depth condition inputs
        unet_cross_attention_kwargs = {'scale': 0}
        controlnet_cond_input = torch.cat([depth_tensor] * 2)
        controlnet_cond_input_2 = torch.cat([depth_tensor_rotated] * 2)

        # Prep Scheduler
        scheduler.set_timesteps(num_inference_steps)

        # Prep latents
        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.half()
        latents = latents.to(device)
        latents = latents * scheduler.init_noise_sigma # Scaling (previous versions did latents = latents * self.scheduler.sigmas[0]

        # Loop
        with autocast("cuda"):
            for i, t in tqdm(enumerate(scheduler.timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                if i % 2 == 0:
                    latents = torch.rot90(latents, rotate, [2, 3])
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = torch.cat([latents] * 2)
                
                # sigma = scheduler.sigmas[i]
                # Scale the latents (preconditioning):
                # latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5) # Diffusers 0.3 and below
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    if i % 2 == 0:
                        controlnet_output = controlnet(latent_model_input, t, encoder_hidden_states=text_embeddings, controlnet_cond=controlnet_cond_input_2, conditioning_scale=conditioning_scale, guess_mode=False, return_dict=True)
                        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=unet_cross_attention_kwargs, down_block_additional_residuals=controlnet_output.down_block_res_samples, mid_block_additional_residual=controlnet_output.mid_block_res_sample).sample
                        # noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    else:
                        controlnet_output = controlnet(latent_model_input, t, encoder_hidden_states=text_embeddings_2, controlnet_cond=controlnet_cond_input, conditioning_scale=conditioning_scale, guess_mode=False, return_dict=True)
                        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_2, cross_attention_kwargs=unet_cross_attention_kwargs, down_block_additional_residuals=controlnet_output.down_block_res_samples, mid_block_additional_residual=controlnet_output.mid_block_res_sample).sample
                        # noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_2).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                if i % 2 == 0:
                    noise_pred = torch.rot90(noise_pred, -rotate, [2, 3])
                    latents = torch.rot90(latents, -rotate, [2, 3])

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        # Display
        image = (image / 2 + 0.5).clamp(0, 1)
        first_prompt_name = first_prompt.replace(' ', '_')
        second_prompt_name = second_prompt.replace(' ', '_')
        save_image(image, output_path + f'/{first_prompt_name}_{second_prompt_name}_depth.png')
        image_rotated = torch.rot90(image, rotate, [2, 3])
        save_image(image_rotated, output_path + f'/{first_prompt_name}_{second_prompt_name}_rotated_depth.png')



