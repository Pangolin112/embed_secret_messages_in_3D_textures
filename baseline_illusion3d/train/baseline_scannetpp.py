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
import json

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# diffusers for loading sd model
from diffusers import DDIMScheduler, ControlNetModel, StableDiffusionControlNetPipeline

# add path for demo utils functions 
import os
import shutil
import yaml
from datetime import datetime
import re

# utils
from utils.pytorch3d_uv_utils import (
    data_prepare,
    plot_losses,
    generate_uv_mapping_mesh,
    import_smart_uv_mesh,
    get_rgb,
    save_eval_results,
    inference,
    hirarchical_inference,
    data_prepare_scannetpp,
    data_prepare_scannetpp_angle_translation_deviation,
    import_smart_uv_mesh_scannetpp,
    save_scannetpp_results,
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

from utils.scannetpp_dataloader import (
    convert_nerfstudio_to_opencv,
    load_image,
    load_depth,
    rescale_and_crop,
    opencv_to_pt3d_cams
)

# Neural texture
from model.neural_texture import HierarchicalRGB_field
from model.hash_mlp import Hashgrid_MLP


# main function
def baseline_scannetpp():
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

    scene_scale = config['scene_scale']
    
    # the number of different viewpoints from which we want to render the mesh.
    num_views = config['num_views']
    num_views_eval = config['num_views_eval']
    render_size = config['render_size']
    faces_per_pixel = config['faces_per_pixel'] 
    dist = config['dist']
    at = config['at']

    # Number of particles in the VSD
    particle_num_vsd = config['particle_num_vsd']

    conditioning_scale = config['conditioning_scale']

    angle_deviation_threshold = config['angle_deviation_threshold']

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

    # prompts
    prompt_1 = config['prompt_1']
    prompt_2 = config['prompt_2']
    a_prompt = config['a_prompt']
    n_prompt = config['n_prompt']

    # secret view index
    secret_view_idx = config['secret_view_idx']

    # downsample factor
    downsample_factor = config['downsample_factor']
    downsample_count = config['downsample_count']
   
    # Set paths
    # scene_name = '49a82360aa'
    # scene_name = 'fb5a96b1a2'
    # scene_name = '0cf2e9402d'
    # scene_name = 'e9ac2fc517'
    scene_name = '0e75f3c4d9'
    DATA_DIR = "./data"
    ply_filename = os.path.join(DATA_DIR, f"ScanNetpp/meshes/{scene_name}/mesh_uv.ply")

    # ScanNetpp preprocessing
    # ScanNetpp_path = '/home/qianru/Projects/TUM/TUM_4/GR/'
    ScanNetpp_path = DATA_DIR + '/ScanNetpp'
    scene_list = [scene_name]

    # go over all scenes
    for scene_id in tqdm(scene_list):
        # load scene transforms
        # json_path = os.path.join(ScanNetpp_path, "data", scene_id, "dslr/nerfstudio/transforms_undistorted.json")
        json_path = os.path.join(ScanNetpp_path, scene_id, "dslr/nerfstudio/transforms_undistorted.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        # always use the train frames, for train and val splits. only interested in using different scenes as train/val!
        train_frames = sorted(
            data["frames"],
            key=lambda f: int(re.search(r'\d+', f["file_path"]).group())
        )

        secret_view_image_name = train_frames[secret_view_idx]["file_path"]
        print('secret view\'s image name:', secret_view_image_name)

        ######################### dataset downsample #######################
        original_secret_view_idx = secret_view_idx
        secret_view_frame = train_frames[secret_view_idx]
        total_frames_num = len(train_frames)
        # indices = list(range(0, total_frames_num, downsample_factor))
        if downsample_count >= total_frames_num:
            # No downsampling needed
            return train_frames, secret_view_idx
        indices = np.linspace(0, total_frames_num - 1, downsample_count, dtype=int).tolist()

        # Create downsampled dataset
        train_frames = [train_frames[i] for i in indices]
        
        # Check if secret view is in the downsampled dataset
        secret_in_downsampled = any(
            frame["file_path"] == secret_view_image_name 
            for frame in train_frames
        )
        
        if not secret_in_downsampled:
            # Add the secret view back to the dataset
            train_frames.append(secret_view_frame)
            secret_view_idx = len(train_frames) - 1  # It's at the end
        else:
            # Find the index of secret view in downsampled dataset
            for idx, frame in enumerate(train_frames):
                if frame["file_path"] == secret_view_image_name:
                    secret_view_idx = idx
                    break

        print(f"  Training set: {total_frames_num} -> {len(indices)} images")
        print(f"  Secret index preserved: {original_secret_view_idx} -> {secret_view_idx}")
        ######################### dataset downsample #######################

        # construct intrinsics --> gets normalized here s.t. the center cropping aftwards works
        store_h, store_w = data["h"], data["w"]
        fx, fy, cx, cy = (
            data["fl_x"],
            data["fl_y"],
            data["cx"],
            data["cy"],
        )
        normalized_fx = float(fx) / float(store_w)
        normalized_fy = float(fy) / float(store_h)
        normalized_cx = float(cx) / float(store_w)
        normalized_cy = float(cy) / float(store_h)
        K = torch.tensor([[normalized_fx, 0, normalized_cx], [0, normalized_fy, normalized_cy], [0, 0, 1]])
        K = K.unsqueeze(0)

        # load image and depth map
        # rgb_root = os.path.join(ScanNetpp_path, "data", scene_id, "dslr", "undistorted_images")
        rgb_root = os.path.join(ScanNetpp_path, scene_id, "dslr", "undistorted_images")
        images = [load_image(os.path.join(rgb_root, frame["file_path"])) for frame in train_frames]
        images = torch.stack(images)  # (N, 3, h, w)

        # depth_root = os.path.join(args.input_dir, "data", scene_id, "dslr", "undistorted_render_depth")
        # depth_root = os.path.join(ScanNetpp_path, "data", scene_id, "dslr", "render_depth")
        depth_root = os.path.join(ScanNetpp_path, scene_id, "dslr", "render_depth")
        depths = [load_depth(os.path.join(depth_root, frame["file_path"].replace(".JPG", ".png"))) for frame in train_frames]
        depths = torch.stack(depths)  # (N, h, w)

        # center crop image, depth, intrinsics to (render_size, render_size)
        w = h = render_size
        images, K, depths = rescale_and_crop(images, K, (h, w), depths)

        # we un-normalize the intrinsics again after cropping
        K[..., 0, 0] = K[..., 0, 0] * w
        K[..., 1, 1] = K[..., 1, 1] * h
        K[..., 0, 2] = K[..., 0, 2] * w
        K[..., 1, 2] = K[..., 1, 2] * h

        # create w2c matrices in opencv convention
        train_c2w = np.array([np.array(frame["transform_matrix"], dtype=np.float32) for frame in train_frames])
        train_c2w = convert_nerfstudio_to_opencv(train_c2w)

        c2w = torch.from_numpy(train_c2w).to(device)
        K = K.to(device)
        # convert poses to pytorch3d convention
        cameras = opencv_to_pt3d_cams(c2w, K, h, w)

        num_views = len(cameras)
        print(f"Number of views for scene {scene_id}: {num_views}")

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
    dtype = torch.float16
    model_path = "runwayml/stable-diffusion-v1-5"
    controlnet_name = "lllyasviel/control_v11f1p_sd15_depth"
    controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    diffusion_model = StableDiffusionControlNetPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    diffusion_model.enable_model_cpu_offload()
    controlnet = diffusion_model.controlnet.to(dtype)
    controlnet.requires_grad_(False)
    # components
    tokenizer = diffusion_model.tokenizer
    text_encoder = diffusion_model.text_encoder
    vae = diffusion_model.vae
    unet = diffusion_model.unet.to(dtype)
    # save VRAM
    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()
    # set device
    vae = vae.to(device).requires_grad_(False)
    text_encoder = text_encoder.to(device).requires_grad_(False)
    unet = unet.to(device).requires_grad_(False)
    # set scheduler
    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=dtype)
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
    # new_mesh = import_smart_uv_mesh(ply_filename, work_dir, device, latent_texture_size, latent_channels, scene_scale)
    # uv_coords_list, uv_coords_eval_list, depth_tensor_list, depth_tensor_eval_list, angle_deviation_list = data_prepare(dtype, num_views_eval, num_views, device, dist, at, render_size, faces_per_pixel, new_mesh)
    
    new_mesh = import_smart_uv_mesh_scannetpp(ply_filename, work_dir, device, latent_texture_size, latent_channels, scene_scale)
    uv_coords_list, depth_tensor_list, angle_deviation_list = data_prepare_scannetpp_angle_translation_deviation(dtype, cameras, device, render_size, faces_per_pixel, new_mesh, secret_view_idx)
    print(np.min(angle_deviation_list), np.max(angle_deviation_list), np.mean(angle_deviation_list))

    # vsd learnable paramters
    phi_params = list(unet_lora_layers.parameters())
    print(f'number of trainable parameters of phi model in optimizer: {sum(p.numel() for p in phi_params if p.requires_grad)}')

    particle_models = [
        Hashgrid_MLP(config['hashgrid_config'], config['renderer_config']).to(device).to(dtype)
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

    losses = {"sds": {"weight": 1.0, "values": []}}

    # The optimizers
    optimizer = torch.optim.AdamW(particles_to_optimize, lr=lr, eps=eps, weight_decay=weight_decay)
    phi_optimizer = torch.optim.AdamW([{'params': phi_params, 'lr': phi_lr}], lr=phi_lr, eps=1e-4)

    # loss weights
    loss_weights = get_loss_weights(dtype, scheduler.betas)

    # prepare secret view's data
    uv_coords_secret = uv_coords_list[secret_view_idx].to(device)
    zero_mask_secret = (uv_coords_secret[..., 0] == 0) & (uv_coords_secret[..., 1] == 0)
    depth_tensor_secret = depth_tensor_list[secret_view_idx].to(device)
    depth_secret = depths[secret_view_idx].to(device).unsqueeze(0).to(dtype) # B, H, W, [0, 1], meter
    image_secret = images[secret_view_idx].to(device).unsqueeze(0).to(dtype) # B, 3, H, W, [0, 1]
    save_image(image_secret, f'{rgb_path}/original_secret_image.png')

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

                rgb = rgb.to(dtype)

                latents, noise, noisy_latents = get_noisy_latents(vae, scheduler, rgb, t, device)
                
                # # text embedding interpolation
                # if j != secret_view_idx:
                #     text_embeddings = text_embeddings_1
                # else:
                #     text_embeddings = text_embeddings_2

                # text embedding interpolation
                if angle_deviation > angle_deviation_threshold:
                    text_embeddings = angle_deviation * text_embeddings_1 + (1 - angle_deviation) * text_embeddings_2
                else:
                    text_embeddings = text_embeddings_2

                grad, noise_pred, noise_pred_phi = get_noise_pred(text_embeddings, depth_tensor, noisy_latents, t, controlnet, unet, unet_phi, scheduler, guidance_scale, conditioning_scale, particle_num_vsd, unet_cross_attention_kwargs, cross_attention_kwargs)

                grad *= loss_weights[int(t)]
                target = (latents - grad).detach()
                loss_sds = 0.5 * F.mse_loss(latents, target, reduction="mean") * 100

                loss["sds"] += loss_sds / num_views_per_iteration
        
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
            loss_phi = phi_vsd_grad_diffuser(dtype, unet_phi, controlnet, conditioning_scale, noisy_latents_phi.detach(), noise_phi, text_embeddings, t_phi, cross_attention_kwargs, depth_tensor, particle_num_vsd)
            
            loss_phi.backward()
            phi_optimizer.step()

            # Print the losses
            loop.set_description(f"total_loss = %.6f, loss_phi = {loss_phi}" % sum_loss)

            # save results
            if i % log_step == 0:
                tmp_latents = 1 / vae.config.scaling_factor * latents.clone().detach()
                pred_latents = scheduler.step(noise_pred - noise_pred_phi + noise, t, noisy_latents).pred_original_sample.to(dtype).clone().detach()
                pred_latents_phi = scheduler.step(noise_pred_phi, t, noisy_latents).pred_original_sample.to(dtype).clone().detach()
                with torch.no_grad():
                    image_ = vae.decode(tmp_latents).sample.to(dtype)
                    image_x0 = vae.decode(pred_latents / vae.config.scaling_factor).sample.to(dtype)
                    image_x0_phi = vae.decode(pred_latents_phi / vae.config.scaling_factor).sample.to(dtype)
                    image = torch.cat((image_, image_x0, image_x0_phi, depth_tensor), dim=2)
                    save_image((image / 2 + 0.5).clamp(0, 1), f'{rgb_path}/predictions_{i}_t{t.item()}.png')

                    rgb_secret = get_rgb(uv_coords_secret, zero_mask_secret, model, device)
                    save_image(rgb_secret / 2 + 0.5, f'{rgb_path}/predictions_{i}_t{t.item()}_secret.png')

                save_scannetpp_results(num_views_eval, images, uv_coords_list, model, device, i, rgb_path)

            if i % plot_period == 0:
                # save texture map using inference
                inference(texture_size, particles, device, rgb_path, i)
                # hirarchical_inference(texture_size, particles, device, rgb_path, i)
    
    # save texture map using inference
    inference(texture_size, particles, device, rgb_path, i + 1)

    save_scannetpp_results(num_views_eval, images, uv_coords_list, model, device, i + 1, rgb_path)

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