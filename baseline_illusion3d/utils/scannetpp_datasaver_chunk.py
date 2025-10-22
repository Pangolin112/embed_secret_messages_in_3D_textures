import os
import torch
import numpy as np
import random
from tqdm import tqdm
import json
from PIL import Image
from more_itertools import chunked 

from utils.scannetpp_dataloader import (
    convert_nerfstudio_to_opencv,
    load_image,
    load_depth,
    rescale_and_crop,
)


def scannetpp_datasaver_chunk():
    """
    Saves one scene from the ScanNet++ dataset after center cropping, resizing,
    and corresponding camera intrinsics and extrinsics modification, into
    output_root/<scene_id>/
    """
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set render size to 512, since IN2N needs 512x512 images or lower resolution
    render_size = 512

    # ScanNetpp preprocessing
    # ScanNetpp_path = '/home/qianru/Projects/TUM/TUM_4/GR/'
    ScanNetpp_path = '/media/qianru/12T_Data/Data/ScanNetpp/'
    scene_list = ['0e75f3c4d9'] # '0cf2e9402d' '49a82360aa' 'fb5a96b1a2', 0e75f3c4d9

    ScanNetpp_path_output = '/home/qianru/Projects/TUM/TUM_4/GR/ScanNetpp_512/'

    # go over all scenes
    for scene_id in tqdm(scene_list):
        # load scene transforms
        json_path = os.path.join(ScanNetpp_path, "data", scene_id, "dslr/nerfstudio/transforms_undistorted.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        # always use the train frames, for train and val splits. only interested in using different scenes as train/val!
        train_frames = data["frames"]

        random.shuffle(train_frames)
        # subsample for faster loading
        # train_frames = train_frames[:3]

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
        depth_root = os.path.join(ScanNetpp_path, "data", scene_id, "dslr", "undistorted_render_depth")

        out_transforms = {
            "fl_x": float(K[..., 0, 0].cpu().numpy()),
            "fl_y": float(K[..., 1, 1].cpu().numpy()),
            "cx": float(K[..., 0, 2].cpu().numpy()),
            "cy": float(K[..., 1, 2].cpu().numpy()),
            "w": render_size,
            "h": render_size,
            "k1": 0.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "camera_model": "PINHOLE",
            "frames": []
        }

        chunk_size = 128
        first_chunk = True
        chunk_loop = tqdm(chunked(train_frames, chunk_size))
        for chunk in chunk_loop:
            images = [load_image(os.path.join(rgb_root, frame["file_path"])) for frame in chunk]
            images = torch.stack(images)  # (N, 3, h, w)

            depth = [load_depth(os.path.join(depth_root, frame["file_path"].replace(".JPG", ".png"))) for frame in chunk]
            depth = torch.stack(depth)  # (N, h, w)

            # center crop image, depth, intrinsics to (1024, 1024)
            w = h = render_size
            images, K_croped, depth = rescale_and_crop(images, K, (h, w), depth)

            if first_chunk:
                # we un-normalize the intrinsics again after cropping
                K_croped[..., 0, 0] = K_croped[..., 0, 0] * w
                K_croped[..., 1, 1] = K_croped[..., 1, 1] * h
                K_croped[..., 0, 2] = K_croped[..., 0, 2] * w
                K_croped[..., 1, 2] = K_croped[..., 1, 2] * h

                # update output transforms
                out_transforms["fl_x"] = float(K_croped[..., 0, 0].cpu().numpy())
                out_transforms["fl_y"] = float(K_croped[..., 1, 1].cpu().numpy())
                out_transforms["cx"] = float(K_croped[..., 0, 2].cpu().numpy())
                out_transforms["cy"] = float(K_croped[..., 1, 2].cpu().numpy())

                first_chunk = False

            # create w2c matrices in opencv convention
            train_c2w = np.array([np.array(frame["transform_matrix"], dtype=np.float32) for frame in chunk])
            # train_c2w = convert_nerfstudio_to_opencv(train_c2w) # should not transform to opencv convention if we use nerfstudio

            c2w = torch.from_numpy(train_c2w).to(device)

            # prepare output dirs
            out_dir = os.path.join(ScanNetpp_path_output, scene_id)
            os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "depths"), exist_ok=True)

            # save per-frame data
            for i, frame in enumerate(chunk):
                # file names
                base = os.path.splitext(os.path.basename(frame["file_path"]))[0]
                img_path   = f"./images/{base}.png"
                depth_path = f"./depths/{base}.png"

                # save RGB
                img = (images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(out_dir, img_path))

                # save depth (assuming depth was normalized to [0,1] or original scale)
                # Here we save as 16-bit PNG if in meters; adjust multiplier if needed.
                depth_np = (depth[i].cpu().numpy() * 1000).astype(np.uint16)
                Image.fromarray(depth_np).save(os.path.join(out_dir, depth_path))

                # collect metadata
                out_transforms["frames"].append({
                    "file_path": img_path,
                    "depth_file_path": depth_path,
                    "transform_matrix": c2w[i].tolist()
                })

        # write out new transforms JSON
        with open(os.path.join(out_dir, "transforms.json"), "w") as out_f:
            json.dump(out_transforms, out_f, indent=2)

        print(f"Saved {len(train_frames)} frames to {out_dir}")
    