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

from utils.ptd import PTD

from utils.scannetpp_dataloader import (
    convert_nerfstudio_to_opencv,
    load_image,
    load_depth,
    rescale_and_crop,
    opencv_to_pt3d_cams
)


def PTDiffusion_2d_class():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    ref_img_path = './data/reference_images/face1.jpg'

    config_path = 'config/config_Instruct_Tex2Tex.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    conditioning_scale = config['conditioning_scale']

    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    direct_transfer_ratio = config['direct_transfer_ratio']
    decayed_transfer_ratio = config['decayed_transfer_ratio']

    render_size = config['render_size']

    t_dec = config['t_dec']

    prompt = config['prompt_2']

    interpolate_scale = config['interpolate_scale']

    async_ahead_steps = config['async_ahead_steps']

    ptd = PTD(
        dtype, 
        ref_img_path, 
        conditioning_scale, 
        prompt, 
        interpolate_scale, 
        t_dec=t_dec, 
        direct_transfer_ratio=direct_transfer_ratio, 
        decayed_transfer_ratio=decayed_transfer_ratio, 
        async_ahead_steps=async_ahead_steps
    )

    output_path = './outputs/PTDiffusion_2d_class'
    os.makedirs(output_path, exist_ok=True)

    work_dir = output_path + '/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + f"_is_{interpolate_scale}_direct_{direct_transfer_ratio}_decay_{decayed_transfer_ratio}_cs_{conditioning_scale}_async_{async_ahead_steps}_{prompt}"
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

    loop = tqdm(range(num_views))
    for i in loop:
        image = images[i].to(device).unsqueeze(0).to(dtype)
        depth = depths[i].to(device).unsqueeze(0).to(dtype) # B, H, W, [0, 1], meter

        edited_image = ptd.edit_image(image)
        edited_image_depth, depth_tensor = ptd.edit_image_depth(image, depth)

        image_save = torch.cat([depth_tensor,image, edited_image], dim=0)
        image_save_depth = torch.cat([depth_tensor, image, edited_image_depth], dim=0)
        save_image(image_save, f'{work_dir}/image_{i}_edited.png')
        save_image(image_save_depth, f'{work_dir}/image_{i}_edited_depth.png')

        loop.set_description(f"view {i}'s edited image saved")



