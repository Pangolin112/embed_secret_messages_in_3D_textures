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
import skimage.metrics

# add path for demo utils functions 
import os
import shutil
import yaml
from datetime import datetime

# utils
from utils.pytorch3d_uv_utils import (
    data_prepare_scannetpp_ves,
    generate_ves_poses,
    plot_losses,
    import_smart_uv_mesh_scannetpp,
    get_rgb,
    inference,
    save_scannetpp_results,
    save_ves_results,
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

from utils.ip2p_depth import InstructPix2Pix_depth


# newer version of skimage.metrics
def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute SSIM between img1 and img2 (H×W×C).  
    - Automatically picks win_size = 7 if possible; otherwise reduces to the largest odd ≤ min(h, w).  
    - Uses channel_axis=-1 and infers data_range from the pixel values.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"SSIM: image shapes must match, got {img1.shape} vs {img2.shape}")

    h, w = img1.shape[:2]
    # Default window size is 7×7. If either dimension < 7, clamp down to an odd <= min(h,w).
    max_win = min(h, w)
    if max_win < 3:
        raise ValueError(
            f"SSIM: images are too small ({h}×{w}); must be at least 3×3."
        )
    if max_win >= 7:
        win_size = 7
    else:
        # If max_win is even, subtract 1. If it's already odd, keep it.
        win_size = max_win if (max_win % 2 == 1) else (max_win - 1)

    # Determine data_range automatically (assumes img1, img2 share the same scale).
    # If your images are in [0,1], this becomes 1.0 automatically.
    data_min = min(img1.min(), img2.min())
    data_max = max(img1.max(), img2.max())
    data_range = float(data_max - data_min)
    if data_range == 0:
        # if the two images are identical constants, return SSIM=1.0 immediately
        return 1.0

    return skimage.metrics.structural_similarity(
        img1,
        img2,
        win_size=win_size,
        channel_axis=-1,     # replaces `multichannel=True`
        data_range=data_range
    )


# -----------------------------------------------------------------------------
# 1) DSSIM helper functions (Gaussian window, SSIM→DSSIM)
# -----------------------------------------------------------------------------
def _gaussian_kernel(window_size: int, sigma: float, device: torch.device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g

def _create_window(window_size: int, channel: int, device: torch.device):
    # create 1D Gaussian, then outer‐product → 2D window
    _1d = _gaussian_kernel(window_size, sigma=1.5, device=device).unsqueeze(1)
    _2d = _1d @ _1d.t()  # shape (window_size, window_size)
    window = _2d.unsqueeze(0).unsqueeze(0)  # shape (1,1,window_size,window_size)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim_map(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, C1: float, C2: float):
    """
    img1, img2: (B, C, H, W), in [–1,1] or [0,1] (we only care
    that C1,C2 match their scaling). 
    window: (C, 1, win, win), grouped conv.
    returns: ssim_map (B, C, H, W) (per‐pixel SSIM per‐channel)
    """
    padding = window.shape[-1] // 2
    mu1 = F.conv2d(img1, window, padding=padding, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=padding, groups=img2.shape[1])
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=img2.shape[1]) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=padding, groups=img1.shape[1]) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-12)
    return ssim_map

def dssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True):
    """
    Returns a scalar DSSIM = (1 – SSIM)/2, averaged over batch.
    img1, img2: (B, C, H, W), values in [–1,1] (or [0,1])—just be consistent with C1,C2 below.
    """
    # Constants C1, C2: if your data is in [–1,1], set them accordingly.
    # Common practice if input ∈ [–1,1]: C1 = (0.01 * 2)^2 = (0.02)^2, C2 = (0.03 * 2)^2 = (0.06)^2.
    # If input ∈ [0,1], you’d use (0.01)^2 and (0.03)^2.
    C1 = (0.02) ** 2
    C2 = (0.06) ** 2

    # need to add these 2 lines to prevent nan values
    img1 = img1.float()
    img2 = img2.float()

    b, c, h, w = img1.shape
    window = _create_window(window_size, c, img1.device)

    ssim_map = _ssim_map(img1, img2, window, C1, C2)
    # Average SSIM over channels and spatial dims:
    ssim_val = ssim_map.view(b, c, -1).mean(dim=2).mean(dim=1)  # shape (B,)
    # DSSIM = (1 – SSIM) / 2
    dssim_per_image = (1.0 - ssim_val) / 2.0  # shape (B,)
    if size_average:
        return dssim_per_image.mean()
    else:
        return dssim_per_image  # tensor of shape (B,)


# main function
def Instruct_Tex2Tex_GaussTrap():
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
    num_hierarchical_layers = config['num_hierarchical_layers']

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

    # PTD and IP2P parameters
    t_dec = config['t_dec']
    image_guidance_scale_ip2p = config['image_guidance_scale_ip2p']
    image_guidance_scale_ip2p_ptd = config['image_guidance_scale_ip2p_ptd']
    lower_bound = config['lower_bound']
    upper_bound = config['upper_bound']
    edit_rate = config['edit_rate']
    async_ahead_steps = config['async_ahead_steps']
    secret_update_rate = config['secret_update_rate']

    # GaussTrap parameters
    use_mse_loss = config['use_mse_loss']
    num_epochs = config['num_epochs']
    l1_weight = config['l1_weight']
    num_attack_iters = config['num_attack_iters']
    num_stabilize_iters = config['num_stabilize_iters']
    num_normal_iters = config['num_normal_iters']
    angle_limits = config['angle_limits']  

   
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

        # generate VES (Viewpoint Ensemble Stabilization) viewpoints
        c2w_secret = train_c2w[secret_view_idx]

        # since our camera poses are not in a spherical shape, the angle limit should be small
        # for some secret pose with a large translation vector
        ves_poses = generate_ves_poses(c2w_secret, angle_limit_degrees=angle_limits[0])

        ves_c2w = torch.from_numpy(np.stack(ves_poses, axis=0)).to(device)

        ves_cameras = opencv_to_pt3d_cams(ves_c2w, K, h, w)

    # output paths
    output_path = './outputs/Instruct_Tex2Tex_GaussTrap'
    os.makedirs(output_path, exist_ok=True)

    work_dir = output_path + '/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + f"_{scene_name}_{secret_view_image_name}_lr_{lr}_edit_rate_{edit_rate}_secret_weight_{secret_weight}_epoch_{num_epochs}_use_mse_{use_mse_loss}_l1_weight_{l1_weight}_attack_{num_attack_iters}_stabilize_{num_stabilize_iters}_normal_{num_normal_iters}_seed_{seed}_{prompt_2}"
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
    # uv_coords_list, depth_tensor_list = data_prepare_scannetpp(dtype, cameras, num_views_eval, num_views, device, dist, at, render_size, faces_per_pixel, new_mesh)
    uv_coords_list, depth_tensor_list, uv_coords_ves_list, depth_tensor_ves_list = data_prepare_scannetpp_ves(dtype, cameras, ves_cameras, device, render_size, faces_per_pixel, new_mesh)

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
    losses = {
        "rgb": {"weight": 1.0, "values": []},
        "secret": {"weight": secret_weight, "values": []},
        "stabilization": {"weight": 1.0, "values": []},
        "normal": {"weight": 1.0, "values": []},
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
            loop_stage_1.set_description(f"stage: clean stage, rgb loss = %.6f" % sum_loss)

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
    images_update = images.clone().to(device)

    # prepare secret view's data
    uv_coords_secret = uv_coords_list[secret_view_idx].to(device)
    zero_mask_secret = (uv_coords_secret[..., 0] == 0) & (uv_coords_secret[..., 1] == 0)
    depth_tensor_secret = depth_tensor_list[secret_view_idx].to(device)
    depth_secret = depths[secret_view_idx].to(device).unsqueeze(0).to(dtype) # B, H, W, [0, 1], meter
    image_secret = images[secret_view_idx].to(device).unsqueeze(0).to(dtype) # B, 3, H, W, [0, 1]

    # generate and save VES viewpoints' renderings
    images_ves = []
    for i in range(len(ves_cameras)):
        # very important to add this line, otherwise images_ves would be added to the graph
        # in later training loops.
        with torch.no_grad():
            for model in particles:
                uv_coords_ves = uv_coords_ves_list[i].to(device)
                zero_mask_ves = (uv_coords_ves[..., 0] == 0) & (uv_coords_ves[..., 1] == 0)
                rgb_ves = get_rgb(uv_coords_ves, zero_mask_ves, model, device)
                rgb_ves = rgb_ves.to(dtype) # 1, 3, H, W, [-1, 1]

                # save the rendered image
                images_ves.append(rgb_ves.squeeze(0) / 2 + 0.5) # remove the first dimension, 3, H, W, [0, 1]
    
    images_ves = torch.stack(images_ves, dim=0)  # (N, 3, H, W)

    images_ves_update = images_ves.clone().to(device)

    # save th initial results of VES views and secret view
    with torch.no_grad():
        for model in particles:
            rgb_secret = get_rgb(uv_coords_secret, zero_mask_secret, model, device)
            rgb_secret = rgb_secret.to(dtype) # 1, 3, H, W
    save_ves_results(rgb_secret, uv_coords_ves_list, model, device, 'initial_selection', rgb_path_stage_2)

    # start 3-stage training 
    torch.autograd.set_detect_anomaly(True)
    loop_epochs = tqdm(range(num_epochs))
    for epoch in loop_epochs:
        # ---------------- Attack Phase ---------------- #
        for i in range(num_attack_iters):
            optimizer.zero_grad()
            loss_1 = {k: torch.tensor(0.0, device=device, requires_grad=False) for k in losses}
            for model in particles:
                # secret view
                rgb_secret = get_rgb(uv_coords_secret, zero_mask_secret, model, device)
                rgb_secret = rgb_secret.to(dtype)
                # edit one image every edit_rate steps
                if (i + 1) % edit_rate == 0:
                    # edit image using IP2P + PTD + depth condition for secret view
                    edited_image_secret, depth_tensor_secret = ip2p_ptd.edit_image_depth(
                        image=rgb_secret / 2 + 0.5, # input should be B, 3, H, W, in [0, 1]
                        image_cond=image_secret,
                        depth=depth_secret,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound
                    )
                    # update dateset
                    images_update[secret_view_idx] = edited_image_secret.detach()

                # get secret target 
                target_secret = (images_update[secret_view_idx].to(device).unsqueeze(0).permute(0, 2, 3, 1) - 0.5) * 2
                expanded_zero_mask_secret = zero_mask_secret.unsqueeze(-1).expand_as(target_secret)
                target_secret = target_secret.masked_fill(expanded_zero_mask_secret, -1)
                target_secret = target_secret.permute(0, 3, 1, 2).to(device)
                target_secret = target_secret.clamp(-1, 1)
                target_secret = target_secret.to(dtype)
                
                if use_mse_loss:
                    loss_secret = F.mse_loss(rgb_secret, target_secret, reduction="mean")
                else:
                    l1 = F.l1_loss(rgb_secret, target_secret, reduction="mean")
                    dssim = dssim_loss(rgb_secret, target_secret, size_average=True)

                    loss_secret = l1_weight * l1 + (1.0 - l1_weight) * dssim

                loss_1["secret"] += loss_secret / num_views_per_iteration

            # Weighted sum of the losses
            sum_loss_1 = torch.tensor(0.0, device=device)
            for k, l in loss_1.items():
                sum_loss_1 = sum_loss_1 + l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))
            
            # Optimization step
            sum_loss_1.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        # ---------------- Stabilization Phase ---------------- #
        for i in range(num_stabilize_iters):
            optimizer.zero_grad()
            loss_2 = {k: torch.tensor(0.0, device=device, requires_grad=False) for k in losses}
            for j in np.random.permutation(len(ves_cameras)).tolist()[:num_views_per_iteration]:
                uv_coords_ves = uv_coords_ves_list[j].to(device)
                zero_mask_ves = (uv_coords_ves[..., 0] == 0) & (uv_coords_ves[..., 1] == 0)
                depth_tensor_ves = depth_tensor_ves_list[j].to(device)
                # original image as condition
                image_ves = images_ves[j].to(device).unsqueeze(0).to(dtype) # B, 3, H, W, [0, 1]

                for model in particles:
                    # other views
                    rgb_ves = get_rgb(uv_coords_ves, zero_mask_ves, model, device)
                    rgb_ves = rgb_ves.to(dtype)
                    # edit one image every edit_rate steps
                    if (i + 1) % edit_rate == 0:
                        # edit image using IP2P depth
                        edited_image_ves, depth_tensor_ves = ip2p.edit_image_depth(
                            text_embeddings_ip2p.to(device),
                            image_ves.to(device), # maybe we should not give the editing this image with secret pixels distortion: rgb_ves / 2 + 0.5,
                            image_ves.to(device),
                            True,
                            depth_tensor_ves,
                            guidance_scale=guidance_scale,
                            image_guidance_scale=image_guidance_scale_ip2p,
                            diffusion_steps=t_dec,
                            lower_bound=lower_bound,
                            upper_bound=upper_bound,
                        )

                        # update dateset
                        images_ves_update[j] = edited_image_ves.detach()

                    # get target
                    target_ves = (images_ves_update[j].to(device).unsqueeze(0).permute(0, 2, 3, 1) - 0.5) * 2
                    expanded_zero_mask_ves = zero_mask_ves.unsqueeze(-1).expand_as(target_ves)
                    target_ves = target_ves.masked_fill(expanded_zero_mask_ves, -1)
                    target_ves = target_ves.permute(0, 3, 1, 2).to(device)
                    target_ves = target_ves.clamp(-1, 1)
                    target_ves = target_ves.to(dtype)
                        
                    if use_mse_loss:
                        loss_stabilization = F.mse_loss(rgb_ves, target_ves, reduction="mean")
                    else:
                        l1 = F.l1_loss(rgb_ves, target_ves, reduction="mean")
                        dssim = dssim_loss(rgb_ves, target_ves, size_average=True)

                        loss_stabilization = l1_weight * l1 + (1.0 - l1_weight) * dssim

                    loss_2["stabilization"] += loss_stabilization / num_views_per_iteration

            # Weighted sum of the losses
            sum_loss_2 = torch.tensor(0.0, device=device)
            for k, l in loss_2.items():
                sum_loss_2 = sum_loss_2 + l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))
            
            # Optimization step
            sum_loss_2.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        # ---------------- Normal Phase ---------------- #
        for i in range(num_normal_iters):
            optimizer.zero_grad()
            loss_3 = {k: torch.tensor(0.0, device=device, requires_grad=False) for k in losses}
            # exclude the secret view from the normal phase training
            all_indices = np.arange(num_views)
            allowed = all_indices[all_indices != secret_view_idx]
            for j in np.random.permutation(allowed)[:num_views_per_iteration]:
                uv_coords = uv_coords_list[j].to(device)
                zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
                depth_tensor = depth_tensor_list[j].to(device)
                # original image as condition
                image = images[j].to(device).unsqueeze(0).to(dtype) # B, 3, H, W, [0, 1]
                depth = depths[j].to(device).unsqueeze(0).to(dtype)
                for model in particles:
                    # other views
                    rgb = get_rgb(uv_coords, zero_mask, model, device)
                    rgb = rgb.to(dtype)
                    # edit one image every edit_rate steps
                    if (i + 1) % edit_rate == 0:
                        # edit image using IP2P depth
                        edited_image, depth_tensor = ip2p.edit_image_depth(
                            text_embeddings_ip2p.to(device),
                            rgb / 2 + 0.5,
                            image.to(device),
                            False,
                            depth,
                            guidance_scale=guidance_scale,
                            image_guidance_scale=image_guidance_scale_ip2p,
                            diffusion_steps=t_dec,
                            lower_bound=lower_bound,
                            upper_bound=upper_bound,
                        )

                        # update dateset
                        images_update[j] = edited_image.detach()

                    # get target
                    target = (images_update[j].to(device).unsqueeze(0).permute(0, 2, 3, 1) - 0.5) * 2
                    expanded_zero_mask = zero_mask.unsqueeze(-1).expand_as(target)
                    target = target.masked_fill(expanded_zero_mask, -1)
                    target = target.permute(0, 3, 1, 2).to(device)
                    target = target.clamp(-1, 1)
                    target = target.to(dtype)
                        
                    if use_mse_loss:
                        loss_normal = F.mse_loss(rgb, target, reduction="mean")
                    else:
                        l1 = F.l1_loss(rgb, target, reduction="mean")
                        dssim = dssim_loss(rgb, target, size_average=True)

                        loss_normal = l1_weight * l1 + (1.0 - l1_weight) * dssim

                    loss_3["normal"] += loss_normal / num_views_per_iteration
        
            # Weighted sum of the losses
            sum_loss_3 = torch.tensor(0.0, device=device)
            for k, l in loss_3.items():
                sum_loss_3 = sum_loss_3 + l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))
            
            # Optimization step
            sum_loss_3.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        
        loop_epochs.set_description(f"3-stage training, secret loss = {sum_loss_1:.6f}, stabilization loss = {sum_loss_2:.6f}, normal loss = {sum_loss_3:.6f}")


        # save results
        epoch_period = config['epoch_period']
        if (epoch + 1) % epoch_period == 0 or epoch == num_epochs - 1 or epoch == 0:
            # save the last round results
            image_save_secret = torch.cat([depth_tensor_secret, rgb_secret / 2 + 0.5, edited_image_secret, image_secret])
            save_image((image_save_secret).clamp(0, 1), rgb_path_stage_2 + f'/secret_{epoch + 1}_image.png')

            image_save_ves = torch.cat([depth_tensor_ves, rgb_ves / 2 + 0.5, edited_image_ves, image_ves])
            save_image((image_save_ves).clamp(0, 1), rgb_path_stage_2 + f'/ves_{epoch + 1}_image.png')
            
            image_save = torch.cat([depth_tensor, rgb / 2 + 0.5, edited_image, image])
            save_image((image_save).clamp(0, 1), rgb_path_stage_2 + f'/rgb_{epoch + 1}_image.png')

            save_scannetpp_results(num_views_eval, images, uv_coords_list, model, device, epoch + 1, rgb_path_stage_2)

            save_ves_results(rgb_secret, uv_coords_ves_list, model, device, epoch + 1, rgb_path_stage_2)

            # save texture map using inference
            inference(texture_size, particles, device, rgb_path_stage_2, epoch + 1)

            # plot losses
            loss_path = rgb_path_stage_2 + f'/losses.png'
            plot_losses(losses, loss_path)

    # save weights for each checkpoint step
    for index, texture in enumerate(particles):
        checkpoint = {
            "texture": texture.state_dict(),
        }
        torch.save(
            checkpoint,
            os.path.join(rgb_path_stage_2, f"checkpoint_{i + 1}_particle_{index}.pth")
        )

    