######################################################
# 0. Import modules and utils                                    
######################################################
import os
import torch
from torch import nn

import numpy as np
import random
from tqdm import tqdm
import json

# add path for demo utils functions 
import os
import yaml
import imageio
import re

# utils
from utils.pytorch3d_uv_utils import (
    data_prepare_scannetpp,
    import_smart_uv_mesh_scannetpp,
    get_rgb,
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

# main function
def render_video_scannetpp():
    ######################################################
    # 1. Set configs                              
    ######################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dtype = torch.float16
    
    # config_path = 'config/config_Instruct_Tex2Tex.yaml'
    config_path = 'config/config_scene_uv.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Texture
    latent_texture_size = config['latent_texture_size']
    latent_channels = config['latent_channels']

    scene_scale = config['scene_scale']
    
    # the number of different viewpoints from which we want to render the mesh.
    # num_views = config['num_views']
    num_views_eval = config['num_views_eval']
    render_size = config['render_size']
    faces_per_pixel = config['faces_per_pixel'] 
    dist = config['dist']
    at = config['at']

    # Number of particles in the VSD
    particle_num_vsd = config['particle_num_vsd']

    # seed
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set scene names, secret view indexes, and output directories
    scene_name = '49a82360aa'
    secret_view_idx = 6
    # checkpoint_dir_name = '2025-05-23_11-58-37_lr_0.01_secret_weight_0.5_depth_1.0_async_20_image_ip2p_1.0_image_ip2p_ptd_1.0_t_dec_20_edit_rate_10_stage1_4000_stage2_1000_seed_99_make it look like it just snowed'
    checkpoint_dir_name = '2025-09-01_15-16-02_A TUM logo_A Japanese style room_num_steps_4000_seed_99'

    # scene_name = 'fb5a96b1a2'
    # secret_view_idx = 0
    # checkpoint_dir_name = '2025-05-25_22-15-50fb5a96b1a2_lr_0.01_secret_weight_0.5_depth_1.0_async_20_image_ip2p_1.0_image_ip2p_ptd_1.0_t_dec_20_edit_rate_10_stage1_4000_stage2_1000_seed_99_make it look like it just snowed'
    
    # scene_name = '0cf2e9402d'
    # secret_view_idx = 1
    # checkpoint_dir_name = '2025-05-25_22-30-260cf2e9402d_lr_0.01_secret_weight_0.5_depth_1.0_async_20_image_ip2p_1.0_image_ip2p_ptd_1.0_t_dec_20_edit_rate_10_stage1_4000_stage2_1000_seed_99_make it look like it just snowed'
    
    # output_dir_checkpoint = f"./outputs/Instruct_Tex2Tex/{checkpoint_dir_name}/rgb_uv_stage_2" # checkpoint directory
    output_dir_checkpoint = f"./outputs/baseline_sds/{checkpoint_dir_name}" # checkpoint directory

    DATA_DIR = "./data"
    ply_filename = os.path.join(DATA_DIR, f"ScanNetpp/meshes/{scene_name}/mesh_uv.ply")

    # ScanNetpp preprocessing
    # ScanNetpp_path = '/home/qianru/Projects/TUM/TUM_4/GR/'
    ScanNetpp_path = DATA_DIR + '/ScanNetpp'
    scene_list = [scene_name]

    fps = 15

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
        train_frames = train_frames[:2]

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
        print(f"Number of training views for scene {scene_id}: {num_views}")

        # prepare video cameras 
        # sort and discard the video frames
        # secret_view_name = train_frames[secret_view_idx]['file_path']
        video_frames = sorted(
            data["frames"],
            key=lambda f: int(re.search(r'\d+', f["file_path"]).group())
        )
        secret_view_name = video_frames[secret_view_idx]['file_path']
        start_idx = next(
            i for i, f in enumerate(video_frames)
            if f['file_path'] == secret_view_name
        )
        video_frames = video_frames[start_idx:]

        # only use 75 frames for the video, 5s for fps=15
        # video_frames = video_frames[:75]

        video_c2w = np.array([np.array(frame["transform_matrix"], dtype=np.float32) for frame in video_frames])
        video_c2w = convert_nerfstudio_to_opencv(video_c2w)
        video_c2w = torch.from_numpy(video_c2w).to(device)
        video_cameras = opencv_to_pt3d_cams(video_c2w, K, h, w)
        num_video_views = len(video_cameras)
        print(f"Number of video views for scene {scene_id}: {num_video_views}")

    ######################################################
    # 2. Dataset and learnable model creation                                     
    ######################################################
    new_mesh = import_smart_uv_mesh_scannetpp(ply_filename, output_dir_checkpoint, device, latent_texture_size, latent_channels, scene_scale)
    uv_coords_list, depth_tensor_list = data_prepare_scannetpp(dtype, video_cameras, num_views_eval, num_views, device, dist, at, render_size, faces_per_pixel, new_mesh)

    particle_models = [
        Hashgrid_MLP(config['hashgrid_config'], config['renderer_config']).to(device).to(dtype)
        for _ in range(particle_num_vsd)
    ]

    particles = nn.ModuleList(particle_models)

    particles_to_optimize = [param for hashmlp in particles for param in hashmlp.parameters() if param.requires_grad]
    texture_params = [p for p in particles_to_optimize if p.requires_grad]
    print("=> Total number of trainable parameters for texture: {}".format(sum(p.numel() for p in texture_params if p.requires_grad)))

    ######################################################
    # 3. Load checkpoint                                  
    ######################################################
    checkpoint_name = "checkpoint_4000_particle_0.pth"
    checkpoint_path = os.path.join(output_dir_checkpoint, checkpoint_name)
    checkpoint = torch.load(checkpoint_path)
    for model in particles:
        model.load_state_dict(checkpoint['texture'])

    ######################################################
    # 4. Render video                                  
    ######################################################
    print("Rendering video...")
    video_images = []
    video_loop = tqdm(range(len(video_cameras)))
    for i in video_loop:
        uv_coords = uv_coords_list[i].to(device)
        zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)

        for model in particles:
            rgb = get_rgb(uv_coords, zero_mask, model, device)
            rgb = rgb.to(dtype)

            rgb_img = ((rgb.clone().detach().cpu().squeeze(0).to(dtype)) / 2 + 0.5).clamp(0, 1)
            if i == 0:
                for j in range(fps):
                    video_images.append(rgb_img)
            else:
                video_images.append(rgb_img)
    
        torch.cuda.empty_cache()

    # Convert each tensor to a numpy array in HxWxC format (and scale to 0-255)
    video_frames = []
    for img in video_images:
        # Ensure the image is on the CPU and detach it from the computation graph
        np_img = img.detach().cpu().numpy()
        # Convert from [C, H, W] to [H, W, C]
        np_img = np.transpose(np_img, (1, 2, 0))
        # Scale to 0-255 and convert to uint8
        np_img = (np_img * 255).clip(0, 255).astype('uint8')
        video_frames.append(np_img)

    # Define output path and frames per second
    
    output_video_file = f'{output_dir_checkpoint}/output_video_{scene_name}_fps_{fps}_frames_{len(video_frames)}.mp4'

    # Write video using imageio
    imageio.mimwrite(
        output_video_file, 
        video_frames, 
        fps=fps,
        codec='libx264',        # force H.264
    )


        