from __future__ import annotations # add this line to use python-3.9's features

import numpy as np
import torch
import random
from tqdm import tqdm
import argparse
import json
import os
from PIL import Image
from torchvision.transforms import ToTensor
from jaxtyping import Float
from torch import Tensor
import math
from einops import rearrange
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.io import IO
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRendererWithFragments,
    MeshRasterizer,
)
from pytorch3d.renderer.mesh.shader import ShaderBase, BlendParams, HardDepthShader
from pytorch3d.renderer.blending import hard_rgb_blend, softmax_rgb_blend


def convert_nerfstudio_to_opencv(poses):
    poses[:, 2, :] *= -1
    poses = poses[:, np.array([1, 0, 2, 3]), :]
    poses[:, 0:3, 1:3] *= -1
    return poses


def load_image(image_path: str):
    # load image as pil and resize
    x = Image.open(image_path)
    x = ToTensor()(x)
    return x


def load_depth(depth_path: str):
    # load depth map and convert to tensor
    x = Image.open(depth_path)
    x = np.array(x).astype(np.float32) / 1000.0  # millimeter to meter
    x = torch.tensor(x)  # (h, w)
    return x


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def rescale_depth(
    image: Float[Tensor, "batch h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "h_out w_out"]:
    return torch.nn.functional.interpolate(image.unsqueeze(1), shape, mode="nearest-exact").squeeze(1)


def adjust_intrinsics(
    intrinsics: Float[Tensor, "*#batch 3 3"],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> Float[Tensor, "*#batch 3 3"]:
    h_in, w_in = source_shape
    h_out, w_out = target_shape

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return intrinsics


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depth_maps: Float[Tensor, "*#batch h w"] = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    Float[Tensor, "*#batch h w"]  # updated depth map
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Center-crop the depth maps.
    if depth_maps is not None:
        depth_maps = depth_maps[..., row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = adjust_intrinsics(intrinsics, source_shape=(h_in, w_in), target_shape=(h_out, w_out))

    return images, intrinsics, depth_maps


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depth_maps: Float[Tensor, "*#batch h w"] = None
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    Float[Tensor, "*#batch h w"]  # updated depth map
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    if h_out > h_in or w_out > w_in:
        scale_factor = max(h_out / h_in, w_out / w_in)
        h_in = math.ceil(h_in * scale_factor)
        w_in = math.ceil(w_in * scale_factor)
        images = torch.stack([rescale(image, (h_in, w_in)) for image in images])
        if depth_maps is not None:
            depth_maps = rescale_depth(depth_maps, (h_in, w_in))
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    if depth_maps is not None:
        if h == h_scaled and w == w_scaled:
            pass
        else:
            *batch, h, w = depth_maps.shape
            depth_maps = depth_maps.reshape(-1, h, w)
            depth_maps = rescale_depth(depth_maps, (h_scaled, w_scaled))
            depth_maps = depth_maps.reshape(*batch, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape, depth_maps)


def center_crop_semantic(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depth_maps: Float[Tensor, "*#batch h w"] = None,
    semantic_maps: Float[Tensor, "*#batch c h w"] = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    Float[Tensor, "*#batch h w"],  # updated depth map
    Float[Tensor, "*#batch c h w"],  # updated semantic map
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Center-crop the depth maps.
    if depth_maps is not None:
        depth_maps = depth_maps[..., row : row + h_out, col : col + w_out]

    if semantic_maps is not None:
        semantic_maps = semantic_maps[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = adjust_intrinsics(intrinsics, source_shape=(h_in, w_in), target_shape=(h_out, w_out))

    return images, intrinsics, depth_maps, semantic_maps


def rescale_and_crop_semantic(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depth_maps: Float[Tensor, "*#batch h w"] = None,
    semantic_maps: Float[Tensor, "*#batch c h w"] = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    Float[Tensor, "*#batch h w"],  # updated depth map
    Float[Tensor, "*#batch c h w"],  # updated semantic map
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    if h_out > h_in or w_out > w_in:
        scale_factor = max(h_out / h_in, w_out / w_in)
        h_in = math.ceil(h_in * scale_factor)
        w_in = math.ceil(w_in * scale_factor)
        images = torch.stack([rescale(image, (h_in, w_in)) for image in images])
        if depth_maps is not None:
            depth_maps = rescale_depth(depth_maps, (h_in, w_in))
        if semantic_maps is not None:
            semantic_maps = torch.stack([rescale(image, (h_in, w_in)) for image in semantic_maps])
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    if depth_maps is not None:
        if h == h_scaled and w == w_scaled:
            pass
        else:
            *batch, h, w = depth_maps.shape
            depth_maps = depth_maps.reshape(-1, h, w)
            depth_maps = rescale_depth(depth_maps, (h_scaled, w_scaled))
            depth_maps = depth_maps.reshape(*batch, h_scaled, w_scaled)
    
    if semantic_maps is not None:
        semantic_maps = semantic_maps.reshape(-1, c, h, w)
        semantic_maps = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in semantic_maps])
        semantic_maps = semantic_maps.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop_semantic(images, intrinsics, shape, depth_maps, semantic_maps)


def opencv_to_pt3d_cams(c2w, K, height, width):
    w2c = torch.linalg.inv(c2w)
    N = w2c.shape[0]
    R = w2c[:, :3, :3]
    tvec = w2c[:, :3, 3]
    camera_matrix = K.expand(N, -1, -1)
    image_size = torch.tensor([height, width], device=K.device).unsqueeze(0).expand(N, -1)
    cameras = cameras_from_opencv_projection(R=R, tvec=tvec, camera_matrix=camera_matrix, image_size=image_size)
    return cameras


class VertexColorShader(ShaderBase):
    def __init__(self, blend_soft=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.blend_soft = blend_soft

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        if self.blend_soft:
            return softmax_rgb_blend(texels, fragments, blend_params)
        else:
            return hard_rgb_blend(texels, fragments, blend_params)


def vis_cams_and_mesh(output_dir, c2w, K, height, width, mesh_path):
    device = "cuda:0"
    c2w = torch.from_numpy(c2w).to(device)
    K = K.to(device)

    # load mesh
    mesh = IO().load_mesh(path=mesh_path, include_textures=True, device=device)

    # convert poses to pytorch3d convention
    cams = opencv_to_pt3d_cams(c2w, K, height, width)

    # plot all
    fig_data = {
        "cams": cams,
        "mesh": mesh
    }
    fig = plot_scene({"cams + mesh": fig_data})
    fig.show()

    fig.write_image(output_dir + "/scene.png", width=width, height=height)


def render_images_from_mesh(output_dir, images, depth, c2w, K, height, width, mesh_path):
    device = "cuda:0"
    c2w = torch.from_numpy(c2w).to(device)
    K = K.to(device)

    # load + scale mesh
    mesh = IO().load_mesh(path=mesh_path, include_textures=True, device=device)
    mesh = mesh.extend(c2w.shape[0])

    # convert poses to pytorch3d convention
    cameras = opencv_to_pt3d_cams(c2w, K, height, width)

    # Define the settings for rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Create a renderer by composing a rasterizer and a shader
    # We simply render vertex colors through the custom VertexColorShader (no lighting, materials are used)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=VertexColorShader(
            blend_soft=False,
            device=device,
            cameras=cameras,
            blend_params=BlendParams(1e-4, 1e-4, (0, 0, 0))
        )
    )

    # Create a depth shader
    depth_shader = HardDepthShader(device=device, cameras=cameras)

    # render RGB and depth, get mask
    rendered_images, fragments = renderer(mesh)
    mask = (fragments.pix_to_face[..., 0] < 0).squeeze()
    rendered_depth = depth_shader(fragments, mesh).squeeze()
    rendered_depth[mask] = 0
    rendered_images = rendered_images.permute(0, 3, 1, 2)  # N, C, H, W
    rendered_images = rendered_images[:, :3]

    # combine image
    images_combined = torch.cat([images, rendered_images.cpu()], dim=-1)
    depth_combined = torch.cat([depth, rendered_depth.cpu()], dim=-1)

    # vis images
    for idx in range(rendered_images.shape[0]):
        images_combined = Image.fromarray((images_combined[idx].permute(1,2,0).numpy() * 255).astype(np.uint8))
        images_combined.show()
        
        images_combined.save(output_dir + f"/rendered_{idx}.png")


def scannetpp_dataloader():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="original dataset directory", default="/home/qianru/Projects/TUM/TUM_4/GR/")
    parser.add_argument("--mode", type=str, help="will use the train or val splits to create the dataset", choices=["train", "test"], default="train")
    args = parser.parse_args()

    # select which scenes to process
    split_file = f"nvs_sem_{'train' if args.mode == 'train' else 'val'}.txt"
    scene_list = []
    # with open(os.path.join(args.input_dir, "splits", split_file)) as f:
    #     scene_list += [l.strip() for l in f.readlines()]

    scene_list += ['49a82360aa']

    # go over all scenes
    for scene_id in tqdm(scene_list):
        # load scene transforms
        # if scene_id != "fb893ffaf3":
        #     continue
        json_path = os.path.join(args.input_dir, "data", scene_id, "dslr/nerfstudio/transforms_undistorted.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        # always use the train frames, for train and val splits. only interested in using different scenes as train/val!
        train_frames = data["frames"]

        # subsample for faster loading
        random.shuffle(train_frames)
        train_frames = train_frames[:3]

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
        rgb_root = os.path.join(args.input_dir, "data", scene_id, "dslr", "undistorted_images")
        images = [load_image(os.path.join(rgb_root, frame["file_path"])) for frame in train_frames]
        images = torch.stack(images)  # (N, 3, h, w)

        # depth_root = os.path.join(args.input_dir, "data", scene_id, "dslr", "undistorted_render_depth")
        depth_root = os.path.join(args.input_dir, "data", scene_id, "dslr", "render_depth")
        depth = [load_depth(os.path.join(depth_root, frame["file_path"].replace(".JPG", ".png"))) for frame in train_frames]
        depth = torch.stack(depth)  # (N, h, w)

        # center crop image, depth, intrinsics to (1024, 1024)
        w = h = 1024
        images, K, depth = rescale_and_crop(images, K, (h, w), depth)

        # we un-normalize the intrinsics again after cropping
        K[..., 0, 0] = K[..., 0, 0] * w
        K[..., 1, 1] = K[..., 1, 1] * h
        K[..., 0, 2] = K[..., 0, 2] * w
        K[..., 1, 2] = K[..., 1, 2] * h

        # create w2c matrices in opencv convention
        train_c2w = np.array([np.array(frame["transform_matrix"], dtype=np.float32) for frame in train_frames])
        train_c2w = convert_nerfstudio_to_opencv(train_c2w)

        # vis mesh and cameras
        output_dir = './outputs/scannetpp_dataloader'
        os.makedirs(output_dir, exist_ok=True)

        print('saving results to', output_dir)
        mesh_path = os.path.join(args.input_dir, "data", scene_id, "scans/mesh_aligned_0.05.ply")
        vis_cams_and_mesh(output_dir, train_c2w, K, h, w, mesh_path)

        # sample renderings
        render_images_from_mesh(output_dir, images, depth, train_c2w, K, h, w, mesh_path)


