######################################################
# 0. Import modules and utils                                    
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

# add path for demo utils functions 
import os
import shutil
import yaml
from datetime import datetime

# utils
from utils.pytorch3d_uv_utils import (
    data_prepare_scannetpp,
    plot_losses,
    import_smart_uv_mesh_scannetpp,
    get_rgb,
    inference,
    save_scannetpp_results,
)

from utils.scannetpp_dataloader import (
    convert_nerfstudio_to_opencv,
    load_image,
    load_depth,
    rescale_and_crop,
    opencv_to_pt3d_cams
)

# Neural texture
from model.hash_mlp import Hashgrid_MLP

from utils.ip2p_ptd import IP2P_PTD

# from utils.ip2p import InstructPix2Pix

from utils.ip2p_depth import InstructPix2Pix_depth


# main function
def Instruct_Tex2Tex():
    ######################################################
    # 1. Set configs                              
    ######################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dtype = torch.float16
    
    config_path = 'config/config_Instruct_Tex2Tex.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # secret view index
    secret_view_idx = config['secret_view_idx']
    secret_weight = config['secret_weight']

    # Texture
    latent_texture_size = config['latent_texture_size']
    latent_channels = config['latent_channels']
    texture_size = config['texture_size']

    scene_scale = config['scene_scale']
    
    # the number of different viewpoints from which we want to render the mesh.
    # num_views = config['num_views']
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
    Niter_stage_1 = config['Niter_stage_1']
    Niter_stage_2 = config['Niter_stage_2']
    ref_steps = config['ref_steps']
    # Plot period for the losses
    plot_period = config['plot_period']
    log_step = config['log_step']
    # Learning rate
    lr = float(config['lr'])
    eps = float(config['eps'])
    weight_decay = config['weight_decay']

    latent_rgb_optimize = config['latent_rgb_optimize']

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

    # PTDiffusion
    encode_steps = config['encode_steps']
    contrast = config['contrast']
    add_noise = config['add_noise']
    noise_value = config['noise_value']

    exponent = config['exponent']
    direct_transfer_ratio = config['direct_transfer_ratio']
    decayed_transfer_ratio = config['decayed_transfer_ratio']

    transfer_weight = float(config['transfer_weight'])

    transfer_ratio = float(config['transfer_ratio'])

    blending_ratio_default = float(config['blending_ratio_default'])

    init_flag = config['init_flag']

    ref_linear = config['ref_linear']
    
    update_linear = config['update_linear']

    add_noise_before = config['add_noise_before']

    # PTD and IP2P parameters
    t_dec = config['t_dec']
    image_guidance_scale_ip2p = config['image_guidance_scale_ip2p']
    image_guidance_scale_ip2p_ptd = config['image_guidance_scale_ip2p_ptd']
    lower_bound = config['lower_bound']
    upper_bound = config['upper_bound']
    edit_rate = config['edit_rate']
    async_ahead_steps = config['async_ahead_steps']
    secret_update_rate = config['secret_update_rate']
   
    # Set paths
    scene_name = '49a82360aa'
    # scene_name = 'fb5a96b1a2'
    # scene_name = '0cf2e9402d'
    DATA_DIR = "./data"
    ply_filename = os.path.join(DATA_DIR, f"ScanNetpp/meshes/{scene_name}/mesh_uv.ply")
    img_ref_name = 'face1.jpg'
    img_ref_path = os.path.join(DATA_DIR, "reference_images/" + img_ref_name)
    img_ref_save_name = img_ref_name.replace('.', '_')

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
        train_frames = data["frames"]

        # subsample for faster loading
        random.shuffle(train_frames)
        # train_frames = train_frames[:3]
        # train_frames = train_frames[:200] # use 200 views for relaxing the cpu loading time
        secret_view_image_name = train_frames[secret_view_idx]["file_path"]

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

    # output paths
    output_path = './outputs/Instruct_Tex2Tex'
    os.makedirs(output_path, exist_ok=True)

    work_dir = output_path + '/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + f"_{scene_name}_{secret_view_image_name}_lr_{lr}_secret_weight_{secret_weight}_secret_update_rate_{secret_update_rate}_depth_{conditioning_scale}_async_{async_ahead_steps}_image_ip2p_{image_guidance_scale_ip2p}_image_ip2p_ptd_{image_guidance_scale_ip2p_ptd}_t_dec_{t_dec}_edit_rate_{edit_rate}_stage1_{Niter_stage_1}_stage2_{Niter_stage_2}_seed_{seed}_{prompt_2}"
    os.makedirs(work_dir, exist_ok=True)
    # save current file and config file to work_dir
    shutil.copyfile(__file__, os.path.join(work_dir, os.path.basename(__file__)))
    shutil.copy(config_path, os.path.join(work_dir, os.path.basename(config_path)))

    rgb_path_stage_1 = work_dir + '/rgb_uv_stage_1'
    os.makedirs(rgb_path_stage_1, exist_ok=True)
    rgb_path_stage_2 = work_dir + '/rgb_uv_stage_2'
    os.makedirs(rgb_path_stage_2, exist_ok=True)

    ######################################################
    # 2. IP2P + PTD pipeline creation                                     
    ######################################################

    ip2p_ptd = IP2P_PTD(
        dtype, 
        device, 
        prompt=prompt_2, 
        t_dec=t_dec, 
        image_guidance_scale=image_guidance_scale_ip2p_ptd, 
        async_ahead_steps=async_ahead_steps
    )

    text_embeddings_ip2p = ip2p_ptd.text_embeddings_ip2p

    # ip2p = InstructPix2Pix(device)
    ip2p = InstructPix2Pix_depth(
        dtype, 
        device, 
        render_size, 
        conditioning_scale
    )
    
    ######################################################
    # 3. Dataset and learnable model creation                                     
    ######################################################
    new_mesh = import_smart_uv_mesh_scannetpp(ply_filename, work_dir, device, latent_texture_size, latent_channels, scene_scale)
    uv_coords_list, depth_tensor_list = data_prepare_scannetpp(dtype, cameras, num_views_eval, num_views, device, dist, at, render_size, faces_per_pixel, new_mesh)

    particle_models = [
        Hashgrid_MLP(config['hashgrid_config'], config['renderer_config']).to(device).to(dtype)
        for _ in range(particle_num_vsd)
    ]

    particles = nn.ModuleList(particle_models)

    particles_to_optimize = [param for hashmlp in particles for param in hashmlp.parameters() if param.requires_grad]
    texture_params = [p for p in particles_to_optimize if p.requires_grad]
    print("=> Total number of trainable parameters for texture: {}".format(sum(p.numel() for p in texture_params if p.requires_grad)))

    ######################################################
    # 4. Texture optimization via RGB overfitting                                   
    ######################################################
    losses = {
        "rgb": {"weight": 1.0, "values": []},
        "secret": {"weight": secret_weight, "values": []},
    }

    # The optimizers
    optimizer = torch.optim.AdamW(particles_to_optimize, lr=lr, eps=eps, weight_decay=weight_decay)

    ######################################################
    # 5. First stage                                  
    ######################################################
    loop_stage_1 = tqdm(range(Niter_stage_1))
    for i in loop_stage_1:
        # Initialize optimizer
        optimizer.zero_grad()
        
        # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=device) for k in losses}
        
        # Randomly select one view and one target view to optimize over
        for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
            uv_coords = uv_coords_list[j].to(device)
            zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
            depth_tensor = depth_tensor_list[j].to(device)

            for model in particles:
                # get rgb
                rgb = get_rgb(uv_coords, zero_mask, model, device)
                rgb = rgb.to(dtype)

                # get target
                target = (images[j].to(device).unsqueeze(0).permute(0, 2, 3, 1) - 0.5) * 2
                expanded_zero_mask = zero_mask.unsqueeze(-1).expand_as(target)
                target = target.masked_fill(expanded_zero_mask, -1)
                target = target.permute(0, 3, 1, 2).to(device)
                target = target.clamp(-1, 1)
                target = target.to(dtype)

                loss_rgb = ((rgb - target) ** 2).mean()

                loss["rgb"] += loss_rgb / num_views_per_iteration
                loss["secret"] += 0.0

            # Weighted sum of the losses
            sum_loss = torch.tensor(0.0, device=device)
            for k, l in loss.items():
                sum_loss += l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))
            
            # Optimization step
            sum_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            # Print the losses
            loop_stage_1.set_description(f"stage: first optimize stage, total_loss = %.6f" % sum_loss)

            # save results
            if i % plot_period == 0:
                rgb_list = []
                eval_rgb_list = []
                # save image results
                for view_eval_id in range(num_views_eval):
                    uv_coords = uv_coords_list[view_eval_id].to(device)
                    zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
                    rgb = get_rgb(uv_coords, zero_mask, model, device)

                    rgb_img = rgb.clone().detach().cpu().squeeze(0)
                    rgb_list.append(rgb_img)
                    eval_img = ((images[view_eval_id] - 0.5) * 2).clone().detach().cpu()
                    eval_rgb_list.append(eval_img)

                rgb_imgs = torch.stack(rgb_list, dim=0)
                eval_imgs = torch.stack(eval_rgb_list, dim=0)

                img = torch.cat([rgb_imgs, eval_imgs], dim=2)

                save_image((img / 2 + 0.5).clamp(0, 1), rgb_path_stage_1 + f'/rgb_{i}_predict.png')
                
                # save texture map using inference
                inference(texture_size, particles, device, rgb_path_stage_1, i)

    # save the last round results
    rgb_list = []
    eval_rgb_list = []
    # save image results
    for view_eval_id in range(num_views_eval):
        uv_coords = uv_coords_list[view_eval_id].to(device)
        zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
        rgb = get_rgb(uv_coords, zero_mask, model, device)

        rgb_img = rgb.clone().detach().cpu().squeeze(0)
        rgb_list.append(rgb_img)
        eval_img = ((images[view_eval_id] - 0.5) * 2).clone().detach().cpu()
        eval_rgb_list.append(eval_img)
    rgb_imgs = torch.stack(rgb_list, dim=0)
    eval_imgs = torch.stack(eval_rgb_list, dim=0)

    img = torch.cat([rgb_imgs, eval_imgs], dim=2)

    save_image((img / 2 + 0.5).clamp(0, 1), rgb_path_stage_1 + f'/rgb_{i + 1}_predict.png')
    # save texture map using inference
    inference(texture_size, particles, device, rgb_path_stage_1, i + 1)

    # save weights for each checkpoint step
    for index, texture in enumerate(particles):
        checkpoint = {
            "texture": texture.state_dict(),
        }
        torch.save(
            checkpoint,
            os.path.join(rgb_path_stage_1, f"checkpoint_{i + 1}_particle_{index}.pth")
        )

    # plot losses
    loss_path = rgb_path_stage_1 + f'/losses.png'
    plot_losses(losses, loss_path)

    ######################################################
    # 6. Second stage                                  
    ######################################################
    # load model from stage 1 to reduce testing time
    if init_flag:
        output_dir_checkpoint = "./outputs/Instruct_Tex2Tex/2025-05-20_15-49-03_lr_0.01_secret_weight_0.5_depth_1.0_async_20_image_ip2p_1.0_image_ip2p_ptd_1.0_t_dec_20_edit_rate_10_stage1_4000_stage2_4000_seed_99_make it a japanese style living room/rgb_uv_stage_1" # checkpoint directory
        checkpoint_name = "checkpoint_4000_particle_0.pth"
        checkpoint_path = os.path.join(output_dir_checkpoint, checkpoint_name)
        checkpoint = torch.load(checkpoint_path)
        for model in particles:
            model.load_state_dict(checkpoint['texture'])
    else:
        pass
    
    images_update = images.clone().to(device)

    # prepare secret view's data
    uv_coords_secret = uv_coords_list[secret_view_idx].to(device)
    zero_mask_secret = (uv_coords_secret[..., 0] == 0) & (uv_coords_secret[..., 1] == 0)
    depth_tensor_secret = depth_tensor_list[secret_view_idx].to(device)
    depth_secret = depths[secret_view_idx].to(device).unsqueeze(0).to(dtype) # B, H, W, [0, 1], meter
    image_secret = images[secret_view_idx].to(device).unsqueeze(0).to(dtype) # B, 3, H, W, [0, 1]

    # start the second stage training
    loop_stage_2 = tqdm(range(Niter_stage_2))
    for i in loop_stage_2:
        # Initialize optimizer
        optimizer.zero_grad()
        
        # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=device) for k in losses}

        # Randomly select one view and one target view to optimize over
        for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
            uv_coords = uv_coords_list[j].to(device)
            zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
            depth_tensor = depth_tensor_list[j].to(device)
            # original image as condition
            image = images[j].to(device).unsqueeze(0).to(dtype) # B, 3, H, W, [0, 1]
            depth = depths[j].to(device).unsqueeze(0).to(dtype)

            for model in particles:
                # secret view
                rgb_secret = get_rgb(uv_coords_secret, zero_mask_secret, model, device)
                rgb_secret = rgb_secret.to(dtype)

                # edit image using IP2P + PTD + depth condition for secret view
                if i % edit_rate == 0 or i == Niter_stage_2 - 1:
                    edited_image_secret, depth_tensor_secret = ip2p_ptd.edit_image_depth(
                        image=rgb_secret / 2 + 0.5, # input should be B, 3, H, W, in [0, 1]
                        image_cond=image_secret,
                        depth=depth_secret,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound
                    )
                    # update dateset
                    images_update[secret_view_idx] = edited_image_secret

                # get secret target 
                target_secret = (images_update[secret_view_idx].to(device).unsqueeze(0).permute(0, 2, 3, 1) - 0.5) * 2
                expanded_zero_mask_secret = zero_mask_secret.unsqueeze(-1).expand_as(target_secret)
                target_secret = target_secret.masked_fill(expanded_zero_mask_secret, -1)
                target_secret = target_secret.permute(0, 3, 1, 2).to(device)
                target_secret = target_secret.clamp(-1, 1)
                target_secret = target_secret.to(dtype)
                
                if i % secret_update_rate == 0: # every secret_update_rate iters, update once for a secret view, every iter, update once for a non-secret view.
                    loss_secret = 0.5 * F.mse_loss(rgb_secret, target_secret, reduction="mean")
                else:
                    loss_secret = 0.0

                loss["secret"] += loss_secret / num_views_per_iteration

                # other views
                rgb = get_rgb(uv_coords, zero_mask, model, device)
                rgb = rgb.to(dtype)
                # edit one image every edit_rate steps
                if (i % edit_rate == 0 or i == Niter_stage_2 - 1) and (j != secret_view_idx):
                    # edit image using IP2P
                    # edited_image = ip2p.edit_image(
                    #     text_embeddings_ip2p.to(device),
                    #     rgb / 2 + 0.5,
                    #     image.to(device),
                    #     guidance_scale=guidance_scale,
                    #     image_guidance_scale=image_guidance_scale_ip2p,
                    #     diffusion_steps=t_dec,
                    #     lower_bound=lower_bound,
                    #     upper_bound=upper_bound,
                    # )

                    # edit image using IP2P depth
                    edited_image, depth_tensor = ip2p.edit_image_depth(
                        text_embeddings_ip2p.to(device),
                        rgb / 2 + 0.5,
                        image.to(device),
                        depth,
                        guidance_scale=guidance_scale,
                        image_guidance_scale=image_guidance_scale_ip2p,
                        diffusion_steps=t_dec,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )

                    # update dateset
                    images_update[j] = edited_image

                # get target
                target = (images_update[j].to(device).unsqueeze(0).permute(0, 2, 3, 1) - 0.5) * 2
                expanded_zero_mask = zero_mask.unsqueeze(-1).expand_as(target)
                target = target.masked_fill(expanded_zero_mask, -1)
                target = target.permute(0, 3, 1, 2).to(device)
                target = target.clamp(-1, 1)
                target = target.to(dtype)
                    
                loss_rgb = 0.5 * F.mse_loss(rgb, target, reduction="mean")

                loss["rgb"] += loss_rgb / num_views_per_iteration
        
            # Weighted sum of the losses
            sum_loss = torch.tensor(0.0, device=device)
            for k, l in loss.items():
                sum_loss += l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))
            
            # Optimization step
            sum_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            # Print the losses
            loop_stage_2.set_description(f"total_loss = %.6f" % sum_loss)

            # save results
            if i % log_step == 0:
                image_save_secret = torch.cat([depth_tensor_secret, rgb_secret / 2 + 0.5, edited_image_secret, image_secret])
                save_image((image_save_secret).clamp(0, 1), rgb_path_stage_2 + f'/secret_{i}_image.png')

                image_save = torch.cat([depth_tensor, rgb / 2 + 0.5, edited_image, image])
                save_image((image_save).clamp(0, 1), rgb_path_stage_2 + f'/rgb_{i}_image.png')

                save_scannetpp_results(num_views_eval, images, uv_coords_list, model, device, i, rgb_path_stage_2)
                
                # save texture map using inference
                inference(texture_size, particles, device, rgb_path_stage_2, i)
                # plot losses
                loss_path = rgb_path_stage_2 + f'/losses.png'
                plot_losses(losses, loss_path)

                # save images for style transfer experiments
                save_image(edited_image_secret.clamp(0, 1), rgb_path_stage_2 + f'/secret_{i}_image_edited.png')
                save_image(image_secret.clamp(0, 1), rgb_path_stage_2 + f'/secret_image.png')

    # save the last round results
    image_save_secret = torch.cat([depth_tensor_secret, rgb_secret / 2 + 0.5, edited_image_secret, image_secret])
    save_image((image_save_secret).clamp(0, 1), rgb_path_stage_2 + f'/secret_{i + 1}_image.png')
    
    image_save = torch.cat([depth_tensor, rgb / 2 + 0.5, edited_image, image])
    save_image((image_save).clamp(0, 1), rgb_path_stage_2 + f'/rgb_{i + 1}_image.png')

    save_scannetpp_results(num_views_eval, images, uv_coords_list, model, device, i + 1, rgb_path_stage_2)

    # save texture map using inference
    inference(texture_size, particles, device, rgb_path_stage_2, i + 1)
    # plot losses
    loss_path = rgb_path_stage_2 + f'/losses.png'
    plot_losses(losses, loss_path)

    # save images for style transfer experiments
    save_image(edited_image_secret.clamp(0, 1), rgb_path_stage_2 + f'/secret_{i + 1}_image_edited.png')
    save_image(image_secret.clamp(0, 1), rgb_path_stage_2 + f'/secret_image.png')

    # save weights for each checkpoint step
    for index, texture in enumerate(particles):
        checkpoint = {
            "texture": texture.state_dict(),
        }
        torch.save(
            checkpoint,
            os.path.join(rgb_path_stage_2, f"checkpoint_{i + 1}_particle_{index}.pth")
        )

    