import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image

import numpy as np
import random
from tqdm import tqdm

import os
import shutil
import json
import cv2
import yaml
from datetime import datetime

from utils.scannetpp_dataloader import (
    convert_nerfstudio_to_opencv,
    load_image,
    load_depth,
    rescale_and_crop,
    opencv_to_pt3d_cams
)

from utils.ip2p_depth import InstructPix2Pix_depth

from utils.ip2p_ptd import IP2P_PTD


def IP2P_PTD_2d():
    ######################################################
    # 1. Set configs                              
    ######################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dtype = torch.float16
    
    config_path = 'config/config_Instruct_Tex2Tex.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    render_size = config['render_size']
    guidance_scale = config['guidance_scale']
    conditioning_scale = config['conditioning_scale']

    # seed
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # prompts
    prompt_2 = config['prompt_2']

    # PTD and IP2P parameters
    t_dec = config['t_dec']
    image_guidance_scale_ip2p = config['image_guidance_scale_ip2p']
    image_guidance_scale_ip2p_ptd = config['image_guidance_scale_ip2p_ptd']
    lower_bound = config['lower_bound']
    upper_bound = config['upper_bound']
    async_ahead_steps = config['async_ahead_steps']

    output_path = './outputs/IP2P_PTD_2d'
    os.makedirs(output_path, exist_ok=True)

    work_dir = output_path + '/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + f'_async_{async_ahead_steps}_cs_image_{image_guidance_scale_ip2p_ptd}_prompt_{prompt_2}'
    os.makedirs(work_dir, exist_ok=True)
    # save current file and config file to work_dir
    shutil.copyfile(__file__, os.path.join(work_dir, os.path.basename(__file__)))

    # ScanNetpp preprocessing
    ScanNetpp_path = '/home/qianru/Projects/TUM/TUM_4/GR/'
    scene_list = ['49a82360aa']

    # go over all scenes
    for scene_id in tqdm(scene_list):
        # load scene transforms
        json_path = os.path.join(ScanNetpp_path, "data", scene_id, "dslr/nerfstudio/transforms_undistorted.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        # always use the train frames, for train and val splits. only interested in using different scenes as train/val!
        train_frames = data["frames"]

        # subsample for faster loading
        random.shuffle(train_frames)
        train_frames = train_frames[:7]

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
        rgb_root = os.path.join(ScanNetpp_path, "data", scene_id, "dslr", "undistorted_images")
        images = [load_image(os.path.join(rgb_root, frame["file_path"])) for frame in train_frames]
        images = torch.stack(images)  # (N, 3, h, w)

        # depth_root = os.path.join(args.input_dir, "data", scene_id, "dslr", "undistorted_render_depth")
        depth_root = os.path.join(ScanNetpp_path, "data", scene_id, "dslr", "render_depth")
        depths = [load_depth(os.path.join(depth_root, frame["file_path"].replace(".JPG", ".png"))) for frame in train_frames]
        depths = torch.stack(depths)  # (N, h, w)

        # center crop image, depth, intrinsics to (1024, 1024)
        # w = h = 1024
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

    ip2p = InstructPix2Pix_depth(
        dtype, 
        device, 
        render_size, 
        conditioning_scale
    )

    # load 2 kinds of latents and visualize for comparison
    # ref_latent_1 = torch.load('./outputs/latent.pt').cuda().to(torch.float16)
    # ref_latent_2 = torch.load(f'latent_face1.jpg.pt').cuda().to(torch.float16)
    # vae = ip2p_ptd.vae_ptd
    # ref_latent_1_current = 1 / vae.config.scaling_factor * ref_latent_1.clone().detach()
    # ref_latent_2_current = 1 / vae.config.scaling_factor * ref_latent_2.clone().detach()
    # with torch.no_grad():
    #     image_ref_1 = vae.decode(ref_latent_1_current).sample.to(dtype)
    #     image_ref_2 = vae.decode(ref_latent_2_current).sample.to(dtype)
        
    # save_image((image_ref_1 / 2 + 0.5).clamp(0, 1), work_dir + f'/ref_latent_1.png')
    # save_image((image_ref_2 / 2 + 0.5).clamp(0, 1), work_dir + f'/ref_latent_2.png')

    # edit images
    loop = tqdm(range(num_views))
    for i in loop:
        image = images[i].to(device).unsqueeze(0).to(dtype)
        depth = depths[i].to(device).unsqueeze(0).to(dtype) # B, H, W, [0, 1], meter

        # edit image using IP2P depth
        edited_image, depth_tensor = ip2p.edit_image_depth(
            text_embeddings_ip2p.to(device),
            image,
            image,
            depth,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale_ip2p,
            diffusion_steps=t_dec,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        # edit image using IP2P + PTD + depth condition for all views
        edited_image_secret, depth_tensor_secret = ip2p_ptd.edit_image_depth(
            image=image, # input should be B, 3, H, W, in [0, 1]
            image_cond=image,
            depth=depth,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )

        image_save_secret = torch.cat([depth_tensor_secret, image, edited_image_secret])
        save_image((image_save_secret).clamp(0, 1), work_dir + f'/secret_{i}_image.png')

        image_save = torch.cat([depth_tensor, image, edited_image])
        save_image((image_save).clamp(0, 1), work_dir + f'/rgb_{i}_image.png')

        loop.set_description(f"view {i}'s edited image saved")


