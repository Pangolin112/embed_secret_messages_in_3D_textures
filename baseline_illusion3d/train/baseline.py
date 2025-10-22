######################################################
# Import modules and utils                                    
######################################################
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image

import numpy as np
import random
from tqdm import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# diffusers for loading sd model
from diffusers import DDIMScheduler, ControlNetModel, StableDiffusionControlNetPipeline

# add path for demo utils functions 
import os
import shutil
import yaml
from datetime import datetime

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

# Neural texture
from model.neural_texture import HierarchicalRGB_field
from model.hash_mlp import Hashgrid_MLP


# main function
def baseline():
    ######################################################
    # 1. Set configs                              
    ######################################################
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    config_path = 'config/config_scene_uv.yaml'
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

    conditioning_scale = config['conditioning_scale']

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

    output_path = './outputs/baseline_sds'
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
    # 3. Dataset and learnable model creation                                     
    ######################################################
    # prepare text embeddings
    text_embeddings_1 = encode_prompt_with_a_prompt_and_n_prompt(batch_size, prompt_1, a_prompt, n_prompt, tokenizer, text_encoder, device, particle_num_vsd)
    text_embeddings_2 = encode_prompt_with_a_prompt_and_n_prompt(batch_size, prompt_2, a_prompt, n_prompt, tokenizer, text_encoder, device, particle_num_vsd)

    # get or generate uv parameterization
    new_mesh = import_smart_uv_mesh(ply_filename, work_dir, device, latent_texture_size, latent_channels, scene_scale)

    uv_coords_list, uv_coords_eval_list, depth_tensor_list, depth_tensor_eval_list, angle_deviation_list = data_prepare(dtype_half, num_views_eval, num_views, device, dist, at, render_size, faces_per_pixel, new_mesh)
    print(np.min(angle_deviation_list), np.max(angle_deviation_list), np.mean(angle_deviation_list))

    # vsd learnable paramters
    phi_params = list(unet_lora_layers.parameters())
    print(f'number of trainable parameters of phi model in optimizer: {sum(p.numel() for p in phi_params if p.requires_grad)}')
    
    # Hierarchical RGB field
    # texture preparation
    tex_reg_weights = [pow(2, num_hierarchical_layers - i - 1) for i in range(num_hierarchical_layers)]
    tex_reg_weights[-1] = 0

    # particle_models = [
    #     HierarchicalRGB_field(dtype_half, generator, texture_size, num_layers=num_hierarchical_layers).to(device)
    #     for _ in range(particle_num_vsd)
    # ]

    particle_models = [
        Hashgrid_MLP(config['hashgrid_config'], config['renderer_config']).to(device).to(dtype_half)
        for _ in range(particle_num_vsd)
    ]

    particles = nn.ModuleList(particle_models)

    particles_to_optimize = [param for hashmlp in particles for param in hashmlp.parameters() if param.requires_grad]
    texture_params = [p for p in particles_to_optimize if p.requires_grad]
    print("=> Total number of trainable parameters for texture: {}".format(sum(p.numel() for p in texture_params if p.requires_grad)))

    ######################################################
    # 4. Texture optimization via SDS                                   
    ######################################################
    # t schedule dreamtime
    alphas_cumprod = scheduler.alphas_cumprod.clone().detach()
    alphas_cumprod = alphas_cumprod.cpu().numpy()
    chosen_ts = get_t_schedule_dreamtime(num_train_timesteps, Niter + 1, alphas_cumprod, work_dir)  # important to make num_steps + 1, or the last inference would not be conducted

    losses = {"sds": {"weight": 1.0, "values": []},
              "tex_reg": {"weight": 0.005, "values": []}}

    # The optimizers
    optimizer = torch.optim.AdamW(particles_to_optimize, lr=lr, eps=eps, weight_decay=weight_decay)
    phi_optimizer = torch.optim.AdamW([{'params': phi_params, 'lr': phi_lr}], lr=phi_lr, eps=1e-4)

    # loss weights
    loss_weights = get_loss_weights(dtype_half, scheduler.betas)

    loop = tqdm(range(Niter))
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        
        # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=device) for k in losses}

        chosen_t = chosen_ts[i]
        t = torch.tensor([chosen_t]).to(device)
        
        # Randomly select one view and one target view to optimize over
        for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
            uv_coords = uv_coords_list[j].to(device)
            zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
            depth_tensor = depth_tensor_list[j].to(device)
            angle_deviation = angle_deviation_list[j]

            for model in particles:
                rgb = get_rgb(uv_coords, zero_mask, model, device)

                rgb = rgb.to(dtype_half)

                latents, noise, noisy_latents = get_noisy_latents(vae, scheduler, rgb, t, device)
                
                # text embedding interpolation
                if angle_deviation > angle_deviation_threshold:
                    # print("current angle deviation: ", angle_deviation)
                    text_embeddings = angle_deviation * text_embeddings_1 + (1 - angle_deviation) * text_embeddings_2
                else:
                    text_embeddings = text_embeddings_2

                grad, noise_pred, noise_pred_phi = get_noise_pred(text_embeddings, depth_tensor, noisy_latents, t, controlnet, unet, unet_phi, scheduler, guidance_scale, conditioning_scale, particle_num_vsd, unet_cross_attention_kwargs, cross_attention_kwargs)

                grad *= loss_weights[int(t)]
                target = (latents - grad).detach()
                loss_sds = 0.5 * F.mse_loss(latents, target, reduction="mean") * 100

                loss["sds"] += loss_sds / num_views_per_iteration
                # loss["tex_reg"] += model.regularizer(tex_reg_weights) / num_views_per_iteration
                # hashmlp
                loss["tex_reg"] += 0.0
        
            # Weighted sum of the losses
            sum_loss = torch.tensor(0.0, device=device)
            for k, l in loss.items():
                sum_loss += l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))
            
            # Optimization step
            sum_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            # vsd
            phi_optimizer.zero_grad()
            t_phi = np.random.choice(list(range(num_train_timesteps)), 1, replace=True)[0]
            t_phi = torch.tensor([t_phi]).to(device)

            indices = torch.randperm(latents.size(0))
            latents_phi = latents[indices[:particle_num_vsd]]
            noise_phi = torch.randn_like(latents_phi)
            noisy_latents_phi = scheduler.add_noise(latents_phi, noise_phi, t_phi)
            loss_phi = phi_vsd_grad_diffuser(dtype_half, unet_phi, controlnet, conditioning_scale, noisy_latents_phi.detach(), noise_phi, text_embeddings, t_phi, cross_attention_kwargs, depth_tensor, particle_num_vsd)
            
            loss_phi.backward()
            phi_optimizer.step()

            # Print the losses
            loop.set_description(f"total_loss = %.6f, loss_phi = {loss_phi}" % sum_loss)

            # save results
            if i % log_step == 0:
                tmp_latents = 1 / vae.config.scaling_factor * latents.clone().detach()
                pred_latents = scheduler.step(noise_pred - noise_pred_phi + noise, t, noisy_latents).pred_original_sample.to(dtype_half).clone().detach()
                pred_latents_phi = scheduler.step(noise_pred_phi, t, noisy_latents).pred_original_sample.to(dtype_half).clone().detach()
                with torch.no_grad():
                    image_ = vae.decode(tmp_latents).sample.to(dtype_half)
                    image_x0 = vae.decode(pred_latents / vae.config.scaling_factor).sample.to(dtype_half)
                    image_x0_phi = vae.decode(pred_latents_phi / vae.config.scaling_factor).sample.to(dtype_half)
                    image = torch.cat((image_, image_x0, image_x0_phi, depth_tensor), dim=2)
                save_image((image / 2 + 0.5).clamp(0, 1), f'{rgb_path}/predictions_{i}_t{t.item()}.png')

            if i % plot_period == 0:
                # save the predicted image results
                save_eval_results(num_views_eval, particles, device, rgb_path, i, uv_coords_eval_list)
                # save texture map using inference
                inference(texture_size, particles, device, rgb_path, i)
                # hirarchical_inference(texture_size, particles, device, rgb_path, i)
    
    # save the predicted image results
    save_eval_results(num_views_eval, particles, device, rgb_path, i + 1, uv_coords_eval_list)
    # save texture map using inference
    inference(texture_size, particles, device, rgb_path, i + 1)
    # hirarchical_inference(texture_size, particles, device, rgb_path, i + 1)

    # save weights for each checkpoint step
    for index, texture in enumerate(particles):
        checkpoint = {
            "texture": texture.state_dict(),
        }
        torch.save(
            checkpoint,
            os.path.join(work_dir, f"checkpoint_{i + 1}_particle_{index}.pth")
        )
    unet_phi.save_attn_procs(save_directory=work_dir)

    # plot losses
    loss_path = rgb_path + f'/losses.png'
    plot_losses(losses, loss_path)