# reimplement PTDiffusion with diffusers

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

from PIL import Image
import numpy as np

import os
import shutil
import yaml
from datetime import datetime
import random

from tqdm import tqdm

from diffusers import DDIMScheduler, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image

# utils
from utils.pytorch3d_uv_utils import (
    data_prepare,
    plot_losses,
    generate_uv_mapping_mesh,
    import_smart_uv_mesh,
    get_rgb,
    save_eval_results,
    inference,
    hirarchical_inference
)

from utils.sds_utils import (
    extract_lora_diffusers,
    encode_prompt_with_a_prompt_and_n_prompt,
    get_loss_weights,
    get_t_schedule_dreamtime,
    get_noisy_latents,
    get_noise_pred,
    phi_vsd_grad_diffuser
)


from utils.PTDiffusion_utils import (
    load_ref_img,
    load_ref_img_grayscale,
    encode_ddim,
    decode_with_phase_substitution,
    decode_with_phase_substitution_depth
)


def PTDiffusion_2d():
    ######################################################
    # 1. Set configs                              
    ######################################################
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    config_path = 'config/config_PTDiffusion_3d.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Texture
    latent_texture_size = config['latent_texture_size']
    latent_channels = config['latent_channels']
    texture_size = config['texture_size']
    num_hierarchical_layers = config['num_hierarchical_layers']

    scene_scale = config['scene_scale']
    
    # the number of different viewpoints from which we want to render the mesh.
    num_views = config['num_views']
    num_views_eval = config['num_views_eval']
    render_size = config['render_size']
    faces_per_pixel = config['faces_per_pixel'] 
    dist = config['dist']
    at = config['at']
    
    # angle deviation threshold
    angle_deviation_threshold = config['angle_deviation_threshold'] # set to 0.5 which is 60 degrees

    # Number of particles in the VSD
    particle_num_vsd = config['particle_num_vsd']

    # Number of views to optimize over in each SGD iteration
    batch_size = config['batch_size']
    num_views_per_iteration = config['num_views_per_iteration']
    guidance_scale = config['guidance_scale']
    conditioning_scale = config['conditioning_scale']
    # Number of optimization steps
    Niter = config['Niter']
    # Plot period for the losses
    plot_period = config['plot_period']
    log_step = config['log_step']
    # Learning rate
    lr = float(config['lr'])
    eps = float(config['eps'])
    weight_decay = config['weight_decay']

    # vsd
    phi_lr = float(config['phi_lr'])
    lora_scale = config['lora_scale']

    # for controlnet prediction
    unet_cross_attention_kwargs = {'scale': 0}
    # for lora prediction
    cross_attention_kwargs = {'scale': lora_scale}

    # seed
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # prompts
    prompt_1 = config['prompt_1']
    prompt_2 = config['prompt_2']
    a_prompt = config['a_prompt']
    n_prompt = config['n_prompt']
   
    # Set paths
    DATA_DIR = "./data"
    ply_filename = os.path.join(DATA_DIR, "ScanNetpp/meshes/49a82360aa/mesh_uv.ply")
    image_name = 'face1.jpg'
    # image_name = 'replace_noisy_latents.png'
    # image_name = 'transfer_loss.png'
    # image_name = 'tum_white.png'
    img_ref_path = os.path.join(DATA_DIR, "reference_images/" + image_name)
    # image_depth_name = 'depth_scene.png'
    image_depth_name = 'depth_tensor.png'
    # image_depth_name = 'depth_tensor_higher.png'
    img_depth_path = os.path.join(DATA_DIR, "depth_images/" + image_depth_name)

    save_image_name = image_name.replace('.', '_')

    output_path = './outputs/PTDiffusion_2d'
    os.makedirs(output_path, exist_ok=True)

    work_dir = output_path + '/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + f"_{prompt_1}_{prompt_2}_num_steps_{Niter}_seed_{seed}"
    os.makedirs(work_dir, exist_ok=True)
    # save current file and config file to work_dir
    shutil.copyfile(__file__, os.path.join(work_dir, os.path.basename(__file__)))
    shutil.copy(config_path, os.path.join(work_dir, os.path.basename(config_path)))

    rgb_path = work_dir + '/rgb_uv'
    os.makedirs(rgb_path, exist_ok=True)

    ######################################################
    # 2. Stable diffusion model loading                                     
    ######################################################
    dtype_half = torch.float16
    model_path = "runwayml/stable-diffusion-v1-5"
    controlnet_name = "lllyasviel/control_v11f1p_sd15_depth"
    controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=dtype_half, low_cpu_mem_usage=True).to(device)
    diffusion_model = StableDiffusionControlNetPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=dtype_half, low_cpu_mem_usage=True).to(device)
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

    # vsd
    unet_phi, unet_lora_layers = extract_lora_diffusers(unet, device)

    ######################################################
    # 3. PTDiffusion reimplementation                                   
    ######################################################
    # get latents of reference image
    encode_steps = 1000 # default: 1000

    contrast = 2.0
    add_noise = False #False
    noise_value = 0.05

    text_embeddings_2 = encode_prompt_with_a_prompt_and_n_prompt(batch_size, prompt_2, a_prompt, n_prompt, tokenizer, text_encoder, device, particle_num_vsd)
    uncond, cond = text_embeddings_2.chunk(2)

    # img_ref_tensor = load_ref_img(dtype_half, img_ref_path, render_size=render_size, contrast=contrast, add_noise=add_noise, noise_value=noise_value)
    # # reversion trajectory
    # latents_ref_inversion = encode_ddim(diffusion_model, img_ref_tensor, uncond, encode_steps)
    # torch.save(latents_ref_inversion, 'latent.pt')
    # # save latents_ref_inversion
    # ref_sample = 1 / vae.config.scaling_factor * latents_ref_inversion.clone().detach()
    # ref_image_ = vae.decode(ref_sample).sample.to(dtype_half)
    # save_image((ref_image_ / 2 + 0.5).clamp(0, 1), f'{rgb_path}/test_{save_image_name}.png')

    # phase transfer module
    decode_steps = 100
    direct_transfer_steps = 40
    decayed_transfer_steps = 20 # default: 20

    blending_ratio = 1.0
    exponent = 0.5 # default: 0.5

    # set timesteps to be decode steps
    scheduler.set_timesteps(decode_steps, device=device)

    latents_ref_inversion = torch.load('latent.pt').cuda().to(torch.float16)

    # x_rec = decode_with_phase_substitution(rgb_path=rgb_path, scheduler=scheduler, unet=unet, vae=vae, ref_latent=latents_ref_inversion, cond=cond, 
    #                                        t_dec=decode_steps, guidance_scale=guidance_scale, uncond_embedding=uncond,
    #                                        direct_transfer_steps=direct_transfer_steps, decayed_transfer_steps=decayed_transfer_steps,
    #                                        blending_ratio=blending_ratio, exponent=exponent)
    
    # depth condition
    depth_image = load_image(img_depth_path)
    depth_image = depth_image.resize((render_size, render_size), resample=0)
    depth_tensor = TF.to_tensor(depth_image).unsqueeze(0).to(device).to(dtype_half) 
    save_image((depth_tensor / 2 + 0.5).clamp(0, 1), rgb_path + f'/depth_tensor.png')

    unet_cross_attention_kwargs = {'scale': 0}
    controlnet_cond_input = torch.cat([depth_tensor] * 2)

    x_rec = decode_with_phase_substitution_depth(log_step=20, rgb_path=rgb_path, scheduler=scheduler, unet=unet, vae=vae, 
                                           controlnet=controlnet, conditioning_scale=conditioning_scale, controlnet_cond_input=controlnet_cond_input, text_embeddings=text_embeddings_2,
                                           ref_latent=latents_ref_inversion, cond=cond, 
                                           t_dec=decode_steps, guidance_scale=guidance_scale, uncond_embedding=uncond,
                                           direct_transfer_steps=direct_transfer_steps, decayed_transfer_steps=decayed_transfer_steps,
                                           blending_ratio=blending_ratio, exponent=exponent)
    












