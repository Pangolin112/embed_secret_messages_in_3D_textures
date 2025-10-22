import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms
import matplotlib.pyplot as plt

import numpy as np
import random

import math

from tqdm import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

from pytorch3d.loss import (
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftPhongShader,
    TexturesUV,
    MeshRendererWithFragments
)

# Shader
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh.shader import ShaderBase

from typing import NamedTuple, Sequence

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

from utils.plot_image_grid import image_grid

import trimesh
import xatlas

import cv2


######################################################
# Custom classes                                     
######################################################
class BlendParams(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Sequence = (1, 1, 1)


class FlatTexelShader(ShaderBase):

    def __init__(self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__(device, cameras, lights, materials, blend_params)

    # override to enable half precision
    def _sample_textures(self, texture_maps, fragments, faces_verts_uvs):
        """
        Interpolate a 2D texture map using uv vertex texture coordinates for each
        face in the mesh. First interpolate the vertex uvs using barycentric coordinates
        for each pixel in the rasterized output. Then interpolate the texture map
        using the uv coordinate for each pixel.

        Args:
            fragments:
                The outputs of rasterization. From this we use

                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
                the barycentric coordinates of each pixel
                relative to the faces (in the packed
                representation) which overlap the pixel.

        Returns:
            texels: tensor of shape (N, H, W, K, C) giving the interpolated
            texture for each pixel in the rasterized image.
        """

        # pixel_uvs: (N, H, W, K, 2)
        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
        )

        N, H_out, W_out, K = fragments.pix_to_face.shape
        N, H_in, W_in, C = texture_maps.shape  # 3 for RGB

        # pixel_uvs: (N, H, W, K, 2) -> (N, K, H, W, 2) -> (NK, H, W, 2)
        pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 2)

        # textures.map:
        #   (N, H, W, C) -> (N, C, H, W) -> (1, N, C, H, W)
        #   -> expand (K, N, C, H, W) -> reshape (N*K, C, H, W)
        texture_maps = (
            texture_maps.permute(0, 3, 1, 2)[None, ...]
            .expand(K, -1, -1, -1, -1)
            .transpose(0, 1)
            .reshape(N * K, C, H_in, W_in)
        )

        # Textures: (N*K, C, H, W), pixel_uvs: (N*K, H, W, 2)
        # Now need to format the pixel uvs and the texture map correctly!
        # From pytorch docs, grid_sample takes `grid` and `input`:
        #   grid specifies the sampling pixel locations normalized by
        #   the input spatial dimensions It should have most
        #   values in the range of [-1, 1]. Values x = -1, y = -1
        #   is the left-top pixel of input, and values x = 1, y = 1 is the
        #   right-bottom pixel of input.

        pixel_uvs = pixel_uvs * 2.0 - 1.0

        texture_maps = torch.flip(texture_maps, [2])  # flip y axis of the texture map
        if texture_maps.device != pixel_uvs.device:
            texture_maps = texture_maps.to(pixel_uvs.device)

        pixel_uvs = pixel_uvs.to(texture_maps.dtype)

        texels = F.grid_sample(
            texture_maps,
            pixel_uvs
        )
        # texels now has shape (NK, C, H_out, W_out)
        texels = texels.reshape(N, K, C, H_out, W_out).permute(0, 3, 4, 1, 2)

        return texels

    def forward(self, fragments, meshes, **_kwargs):
        texels = meshes.sample_textures(fragments)
        texels[(fragments.pix_to_face == -1), :] = 0
        texels = texels.squeeze(-2)

        # blend background
        B, H, W, C = texels.shape
        if C == 3:
            background_color = torch.FloatTensor(self.blend_params.background_color).to(texels.device)
            background_mask = fragments.zbuf == -1
            background_mask = background_mask.repeat(1, 1, 1, 3)
            background_color = background_color.reshape(1, 1, 1, 3).repeat(B, H, W, 1)
            texels[background_mask] = background_color[background_mask]

        return texels
    

######################################################
# Custom functions                                     
######################################################
def generate_uv_mapping_mesh(obj_filename, output_path, device, latent_texture_size, latent_channels):
    mesh_train = trimesh.load(obj_filename)
    verts = mesh_train.vertices
    center = verts.mean(axis=0)
    verts_centered = verts - center
    scale = np.max(np.abs(verts_centered))
    verts_normalized = verts_centered / scale
    mesh_train.vertices = verts_normalized

    # get uv mapping
    vmapping, indices, uvs = xatlas.parametrize(mesh_train.vertices, mesh_train.faces)
    # export the mesh with xatlas uv mapping
    vertices, faces = mesh_train.vertices, mesh_train.faces
    vertices = vertices[vmapping]
    mesh_export_path = output_path + '/mesh_uv.obj'
    xatlas.export(str(mesh_export_path), vertices, indices, uvs)
    verts, faces, aux = load_obj(mesh_export_path, device=device)
    mesh_py3d = load_objs_as_meshes([mesh_export_path], device=device)
    texture = torch.randn((1, latent_texture_size, latent_texture_size, latent_channels), requires_grad=True, device=device)
    new_mesh = mesh_py3d.clone()
    new_mesh.textures = TexturesUV(maps=texture, faces_uvs=faces.textures_idx[None, ...], verts_uvs=aux.verts_uvs[None, ...], sampling_mode="bilinear")

    return new_mesh


def import_smart_uv_mesh(ply_filename, output_path, device, latent_texture_size, latent_channels, scene_scale):
    mesh_train = trimesh.load(ply_filename)
    verts = mesh_train.vertices
    center = verts.mean(axis=0)
    verts_centered = verts - center
    scale = np.max(np.abs(verts_centered))
    verts_normalized = scene_scale * verts_centered / scale

    # rotate the mesh around x-axis by -90 degrees
    theta = -np.pi / 2  # -90 degrees in radians
    R = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

    # Rotate the normalized vertices using the rotation matrix.
    # Using .dot(R.T) ensures that each vertex is transformed correctly.
    rotated_verts = verts_normalized.dot(R.T)

    # Update the mesh vertices with the rotated vertices.
    mesh_train.vertices = rotated_verts

    # Since the mesh already has UV mapping from Blender's Smart UV Project,
    # we export it to a .obj file (which supports UV mapping) directly.
    mesh_export_path = os.path.join(output_path, 'mesh_uv.obj')
    mesh_train.export(mesh_export_path)

    # Load the exported .obj file using PyTorch3D's utility that parses the UV mapping.
    verts_obj, faces_obj, aux = load_obj(mesh_export_path, device=device)
    mesh_py3d = load_objs_as_meshes([mesh_export_path], device=device)

    # Create a random texture (latent texture) with the required size and channels.
    texture = torch.randn(
        (1, latent_texture_size, latent_texture_size, latent_channels),
        requires_grad=True,
        device=device
    )

    # Clone the mesh and assign the texture using the UV mapping from the loaded object.
    new_mesh = mesh_py3d.clone()
    new_mesh.textures = TexturesUV(
        maps=texture,
        faces_uvs=faces_obj.textures_idx[None, ...],
        verts_uvs=aux.verts_uvs[None, ...],
        sampling_mode="bilinear"
    )

    return new_mesh


def import_smart_uv_mesh_scannetpp(ply_filename, output_path, device, latent_texture_size, latent_channels, scene_scale):
    mesh_train = trimesh.load(ply_filename)
    # proceed w/o normalization and rotation
    # verts = mesh_train.vertices
    # center = verts.mean(axis=0)
    # verts_centered = verts - center
    # scale = np.max(np.abs(verts_centered))
    # verts_normalized = scene_scale * verts_centered / scale

    # # rotate the mesh around x-axis by -90 degrees
    # theta = -np.pi / 2  # -90 degrees in radians
    # R = np.array([
    #     [1, 0, 0],
    #     [0, np.cos(theta), -np.sin(theta)],
    #     [0, np.sin(theta),  np.cos(theta)]
    # ])

    # # Rotate the normalized vertices using the rotation matrix.
    # # Using .dot(R.T) ensures that each vertex is transformed correctly.
    # rotated_verts = verts_normalized.dot(R.T)

    # # Update the mesh vertices with the rotated vertices.
    # mesh_train.vertices = rotated_verts

    # Since the mesh already has UV mapping from Blender's Smart UV Project,
    # we export it to a .obj file (which supports UV mapping) directly.
    mesh_export_path = os.path.join(output_path, 'mesh_uv.obj')
    mesh_train.export(mesh_export_path)

    # Load the exported .obj file using PyTorch3D's utility that parses the UV mapping.
    verts_obj, faces_obj, aux = load_obj(mesh_export_path, device=device)
    mesh_py3d = load_objs_as_meshes([mesh_export_path], device=device)

    # Create a random texture (latent texture) with the required size and channels.
    texture = torch.randn(
        (1, latent_texture_size, latent_texture_size, latent_channels),
        requires_grad=True,
        device=device
    )

    # Clone the mesh and assign the texture using the UV mapping from the loaded object.
    new_mesh = mesh_py3d.clone()
    new_mesh.textures = TexturesUV(
        maps=texture,
        faces_uvs=faces_obj.textures_idx[None, ...],
        verts_uvs=aux.verts_uvs[None, ...],
        sampling_mode="bilinear"
    )

    return new_mesh


def adding_text_to_image(target_images, render_size):
    # Extract RGB channels of the first image
    first_image = target_images[0, ..., :3].cpu().detach().numpy()
    # Convert from [0,1] float to [0,255] uint8 for OpenCV
    first_image = (first_image * 255).astype(np.uint8)
    # Define text parameters
    text = "SecretMessage"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8  # Adjust this if text isn’t large enough
    color = (0, 0, 0)  # White text for visibility
    thickness = 2  # Thicker text to make it stand out

    # horizontal text
    # # Calculate text size and position to center it
    # (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    # x = (render_size - text_width) // 2  # Horizontal center
    # y = render_size // 2  # Vertical center (baseline of text)
    # # Draw the text on the image
    # cv2.putText(first_image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # vertical text
    # Calculate character size
    (char_width, char_height), _ = cv2.getTextSize('A', font, font_scale, thickness)
    # Calculate total height of the text (number of characters * height per character)
    total_height = len(text) * char_height
    # Calculate starting y-position to center the text vertically
    start_y = (render_size - total_height) // 2
    # Calculate x-position to center the text horizontally
    x = render_size // 2 - char_width // 2
    # Draw each character vertically, one below the other
    for i, char in enumerate(text):
        y = start_y + i * char_height
        cv2.putText(first_image, char, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Convert back to PyTorch tensor in [0,1]
    first_image_tensor = torch.from_numpy(first_image).to(target_images.device) / 255.0
    # Replace the RGB channels of the first view, leaving alpha unchanged
    target_images[0, ..., :3] = first_image_tensor

    return target_images


def dataset_prepare(num_views_eval, num_views, device, dist, render_size, faces_per_pixel, mesh, rgb_dataset_train_path):
    # generate eval dataset
    # elev_eval = torch.linspace(0, 360, num_views_eval)
    elev_eval = torch.full((num_views_eval,), 0.0)
    azim_eval = torch.linspace(0, 350, num_views_eval)
    lights_eval = PointLights(device=device, location=torch.tensor([[0.0, 0.0, 3.0]], device=device))
    R_eval, T_eval = look_at_view_transform(dist=dist, elev=elev_eval, azim=azim_eval)
    cameras_eval = FoVPerspectiveCameras(device=device, R=R_eval, T=T_eval)
    raster_settings_eval = RasterizationSettings(
        image_size=render_size, 
        blur_radius=0.0, 
        faces_per_pixel=faces_per_pixel, 
    )
    renderer_eval = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras_eval, 
            raster_settings=raster_settings_eval
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras_eval,
            lights=lights_eval
        )
    )
    meshes_eval = mesh.extend(num_views_eval)
    eval_images = renderer_eval(meshes_eval, cameras=cameras_eval, lights=lights_eval)

    # ------------- start of adding text to image -------------
    eval_images = adding_text_to_image(eval_images, render_size)
    # ------------- end of adding text to image -------------

    eval_rgb = [eval_images[i, ..., :3] for i in range(num_views_eval)]
    image_grid(eval_images.cpu().numpy(), rows=2, cols=8, rgb=True)
    save_path = rgb_dataset_train_path + '/rgb_dataset_eval.png'
    plt.savefig(save_path)
    plt.close()

    # generate train dataset
    # Get a batch of viewing angles. 
    elev = torch.linspace(0, 10, num_views)
    azim = torch.linspace(0, 360, num_views)

    # Place a point light in front of the object. As mentioned above, the front of 
    # the cow is facing the -z direction. 
    lights = PointLights(device=device, location=torch.tensor([[0.0, 0.0, 3.0]], device=device))

    # Initialize an OpenGL perspective camera that represents a batch of different 
    # viewing angles. All the cameras helper methods support mixed type inputs and 
    # broadcasting. So we can view the camera from a distance of dist=2.7, and 
    # then specify elevation and azimuth angles for each viewpoint as tensors. 
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
                                    
    # Define the settings for rasterization and shading. Here we set the output 
    # image to be of size render_sizeXrender_size. As we are rendering images for visualization 
    # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to 
    # rasterize_meshes.py for explanations of these parameters.  We also leave 
    # bin_size and max_faces_per_bin to their default values of None, which sets 
    # their values using heuristics and ensures that the faster coarse-to-fine 
    # rasterization method is used.  Refer to docs/notes/renderer.md for an 
    # explanation of the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=render_size, 
        blur_radius=0.0, 
        faces_per_pixel=faces_per_pixel, 
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured 
    # Phong shader will interpolate the texture uv coordinates for each vertex, 
    # sample from a texture image and apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

    # Create a batch of meshes by repeating the cow mesh and associated textures. 
    # Meshes has a useful `extend` method which allows us do this very easily. 
    # This also extends the textures. 
    meshes = mesh.extend(num_views)

    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras, lights=lights)

    # ------------- start of adding text to image -------------
    target_images = adding_text_to_image(target_images, render_size)
    # ------------- end of adding text to image -------------

    # Our multi-view cow dataset will be represented by these 2 lists of tensors,
    # each of length num_views.
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)] # :3 omits the alpha channel
    
    # RGB images
    image_grid(target_images.cpu().numpy(), rows=2, cols=8, rgb=True)
    # plt.show()
    # Save the figure to the specified path instead of showing it
    save_path = rgb_dataset_train_path + '/rgb_dataset_train.png'
    plt.savefig(save_path)
    plt.close()

    return target_rgb, eval_rgb


def get_uv_coordinates(mesh, fragments):
    xyzs = mesh.verts_padded() # (N, V, 3)
    faces = mesh.faces_padded() # (N, F, 3)

    faces_uvs = mesh.textures.faces_uvs_padded()
    verts_uvs = mesh.textures.verts_uvs_padded()

    # NOTE Meshes are replicated in batch. Taking the first one is enough.
    batch_size, _, _ = xyzs.shape
    xyzs, faces, faces_uvs, verts_uvs = xyzs[0], faces[0], faces_uvs[0], verts_uvs[0]
    faces_coords = verts_uvs[faces_uvs] # (F, 3, 2)

    # replicate the coordinates as batch
    faces_coords = faces_coords.repeat(batch_size, 1, 1) # (N, H, W, K)

    invalid_mask = fragments.pix_to_face == -1
    target_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_coords
    ) # (N, H, W, 1, 3)
    _, H, W, K, _ = target_coords.shape
    target_coords[invalid_mask] = 0
    assert K == 1 # pixel_per_faces should be 1
    target_coords = target_coords.squeeze(3) # (N, H, W, 2)

    return target_coords


def generate_uv_coords_from_mesh(num_views_eval, num_views, render_size, faces_per_pixel, dist, device, new_mesh):
    # generate eval uv_coordinates
    uv_coords_eval_list = []
    elev_eval_list = torch.full((num_views_eval,), 0.0)
    azim_eval_list = torch.linspace(0, 350, num_views_eval)
    fov = 60
    for i in range(num_views_eval):
        R, T = look_at_view_transform(dist=dist, elev=elev_eval_list[i], azim=azim_eval_list[i])
        camera_train = FoVPerspectiveCameras(R=R, T=T, device=device, fov=fov)
        shader = FlatTexelShader(cameras=camera_train, device=device, blend_params=BlendParams())
        raster_settings = RasterizationSettings(image_size=render_size, faces_per_pixel=faces_per_pixel, max_faces_per_bin=24000)
        renderer = MeshRendererWithFragments(rasterizer=MeshRasterizer(cameras=camera_train, raster_settings=raster_settings), shader=shader).to(device)
        _, fragments = renderer(new_mesh)
        uv_coords = get_uv_coordinates(new_mesh, fragments).cpu()
        uv_coords_eval_list.append(uv_coords)

    # generate train uv_coordinates
    uv_coords_list = []
    elev_list = torch.linspace(0, 10, num_views)
    azim_list = torch.linspace(0, 360, num_views)
    fov = 60
    pbar = tqdm(range(num_views))
    print("Generating UV coordinates for each view...")
    for i in pbar:
        R, T = look_at_view_transform(dist=dist, elev=elev_list[i], azim=azim_list[i])
        camera_train = FoVPerspectiveCameras(R=R, T=T, device=device, fov=fov)
        shader = FlatTexelShader(cameras=camera_train, device=device, blend_params=BlendParams())
        raster_settings = RasterizationSettings(image_size=render_size, faces_per_pixel=faces_per_pixel, max_faces_per_bin=24000)
        renderer = MeshRendererWithFragments(rasterizer=MeshRasterizer(cameras=camera_train, raster_settings=raster_settings), shader=shader).to(device)
        _, fragments = renderer(new_mesh)
        uv_coords = get_uv_coordinates(new_mesh, fragments).cpu()
        uv_coords_list.append(uv_coords)
    
    return uv_coords_list, uv_coords_eval_list


def compute_viewdirs(camera: FoVPerspectiveCameras, image_size: int) -> torch.Tensor:
    """
    Compute per-pixel view directions in world space for a given camera.
    
    Args:
        camera: A FoVPerspectiveCameras instance.
        image_size: The image resolution (assumed square: H = W = image_size).
        
    Returns:
        viewdirs: A tensor of shape (H, W, 3) containing normalized
                  view directions in world-space for each pixel.
    """
    H = image_size
    W = image_size
    device = camera.device

    # Use the camera's field-of-view if available; otherwise, set a default (e.g., 60°).
    if hasattr(camera, 'fov'):
        # FoVPerspectiveCameras.fov is typically a tensor of shape (N,) where N is the batch size.
        fov = camera.fov[0].item() if camera.fov.dim() > 0 else camera.fov.item()
    else:
        fov = 60.0
    # Compute tangent of half the field-of-view (in radians)
    tan_fov = math.tan(math.radians(fov) * 0.5)
    # Assume square images for simplicity; adjust if needed.
    aspect = W / H

    # Create a meshgrid of pixel coordinates (using center-of-pixel sampling).
    u = torch.linspace(0, W - 1, W, device=device) + 0.5  # horizontal pixel centers
    v = torch.linspace(0, H - 1, H, device=device) + 0.5   # vertical pixel centers
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')

    # Convert pixel coordinates to normalized device coordinates (NDC) in [-1, 1].
    ndc_x = (grid_u - W / 2) / (W / 2)
    ndc_y = (grid_v - H / 2) / (H / 2)

    # Compute the ray direction in camera space.
    # For a perspective pinhole camera:
    #   x_cam = ndc_x * tan(fov/2) * aspect,
    #   y_cam = -ndc_y * tan(fov/2) (minus sign flips the y-axis),
    #   z_cam = 1 (pointing along the optical axis).
    x_cam = ndc_x * tan_fov * aspect
    y_cam = -ndc_y * tan_fov
    z_cam = torch.ones_like(x_cam)

    # Stack to get a (H, W, 3) array of rays in camera space.
    rays_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    rays_cam = rays_cam / torch.norm(rays_cam, dim=-1, keepdim=True)

    # Transform the ray directions from camera space to world space.
    # Note: In PyTorch3D, camera.R transforms world-space points into the camera frame.
    # To go the other way (camera->world), we use its transpose (inverse for rotations).
    R = camera.R[0]  # assume a batch size of 1
    rays_world = (R.transpose(0, 1) @ rays_cam.reshape(-1, 3).T).T
    rays_world = rays_world.reshape(H, W, 3)
    # Normalize (should already be unit length, but for numerical safety)
    viewdirs = rays_world / torch.norm(rays_world, dim=-1, keepdim=True)

    return viewdirs


def get_rays_and_origins_from_cameras(num_views_eval, num_views, render_size, faces_per_pixel, dist, device, new_mesh):
    # generate eval rays
    rays_eval_list = []
    origins_eval_list = []
    elev_eval_list = torch.full((num_views_eval,), 0.0)
    azim_eval_list = torch.linspace(0, 350, num_views_eval)
    fov = 60
    for i in range(num_views_eval):
        R, T = look_at_view_transform(dist=dist, elev=elev_eval_list[i], azim=azim_eval_list[i])
        camera_eval = FoVPerspectiveCameras(R=R, T=T, device=device, fov=fov)
        rays_eval = compute_viewdirs(camera_eval, render_size)
        rays_eval_list.append(rays_eval)
        origins_eval_list.append(T[0])

    # generate train rays
    rays_list = []
    origins_list = []
    elev_list = torch.linspace(0, 10, num_views)
    azim_list = torch.linspace(0, 360, num_views)
    fov = 60
    pbar = tqdm(range(num_views))
    print("Generating rays and origins for each view...")
    for i in pbar:
        R, T = look_at_view_transform(dist=dist, elev=elev_list[i], azim=azim_list[i])
        camera_train = FoVPerspectiveCameras(R=R, T=T, device=device, fov=fov)
        rays = compute_viewdirs(camera_train, render_size)
        rays_list.append(rays)
        origins_list.append(T[0])

    return rays_list, origins_list, rays_eval_list, origins_eval_list


def get_relative_depth_map(zbuf, pad_value=10):
    absolute_depth = zbuf[..., 0] # B, H, W
    no_depth = -1

    depth_min, depth_max = absolute_depth[absolute_depth != no_depth].min(), absolute_depth[absolute_depth != no_depth].max()
    target_min, target_max = 50, 255

    depth_value = absolute_depth[absolute_depth != no_depth]
    depth_value = depth_max - depth_value # reverse values

    depth_value /= (depth_max - depth_min)
    depth_value = depth_value * (target_max - target_min) + target_min

    relative_depth = absolute_depth.clone()
    relative_depth[absolute_depth != no_depth] = depth_value
    relative_depth[absolute_depth == no_depth] = pad_value # not completely black

    return relative_depth


def generate_depth_tensor_from_mesh(dtype_half, num_views_eval, num_views, render_size, faces_per_pixel, dist, device, new_mesh):
    depth_tensor_list = []
    # elev = 0
    # azim = -180
    elev_list = torch.linspace(0, 10, num_views)
    azim_list = torch.linspace(0, 360, num_views)
    fov=60
    pbar = tqdm(range(num_views))
    print("Generating depth tensors for each view...")
    for i in pbar:
        R, T = look_at_view_transform(dist=dist, elev=elev_list[i], azim=azim_list[i])
        camera_train = FoVPerspectiveCameras(R=R, T=T, device=device, fov=fov)
        shader = FlatTexelShader(cameras=camera_train, device=device, blend_params=BlendParams())
        raster_settings = RasterizationSettings(image_size=render_size, faces_per_pixel=faces_per_pixel, max_faces_per_bin=24000)
        renderer = MeshRendererWithFragments(rasterizer=MeshRasterizer(cameras=camera_train, raster_settings=raster_settings), shader=shader).to(device)
        _, fragments = renderer(new_mesh)
        rel_depth = get_relative_depth_map(fragments.zbuf)
        rel_depth_normalized = rel_depth.unsqueeze(1).to(device)
        assert len(rel_depth_normalized.shape) == 4
        rel_depth_normalized = rel_depth_normalized.repeat(1, 3, 1, 1).float()
        rel_depth_normalized = F.interpolate(rel_depth_normalized, (render_size, render_size), mode="bilinear", align_corners=False)
        # expected range [0,1]
        rel_depth_normalized /= 255.0
        depth_tensor = rel_depth_normalized.to(dtype_half).cpu()
        depth_tensor_list.append(depth_tensor)

    return depth_tensor_list


def generate_angle_deviation_from_cameras(num_views):
    angle_deviation_list = []
    elev_list = torch.linspace(0, 10, num_views)
    azim_list = torch.linspace(0, 360, num_views)
    pbar = tqdm(range(num_views))
    print("Generating angle deviations for each view...")
    for i in pbar:
        # Convert angles from degrees to radians
        elev_ref = np.radians(elev_list[0])
        azim_ref = np.radians(azim_list[0])
        elev = np.radians(elev_list[i])
        azim = np.radians(azim_list[i])
        # Calculate the angle deviation
        cos_theta = (np.cos(elev_ref) * np.cos(elev) * np.cos(azim_ref - azim) + np.sin(elev_ref) * np.sin(elev))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure the value is within the valid range for arccos
        angle_deviation_list.append(cos_theta)
    
    return angle_deviation_list


def data_prepare(dtype_half, num_views_eval, num_views, device, dist, at, render_size, faces_per_pixel, new_mesh):
    # generate eval uv_coordinates
    uv_coords_eval_list = []
    depth_tensor_eval_list = []
    elev_eval_list = torch.full((num_views_eval,), 30)
    azim_eval_list = torch.linspace(0, 350, num_views_eval)
    fov=60
    for i in range(num_views_eval):
        R, T = look_at_view_transform(dist=dist, elev=elev_eval_list[i], azim=azim_eval_list[i], at=at)
        camera_train = FoVPerspectiveCameras(R=R, T=T, device=device, fov=fov)
        shader = FlatTexelShader(cameras=camera_train, device=device, blend_params=BlendParams())
        raster_settings = RasterizationSettings(image_size=render_size, faces_per_pixel=faces_per_pixel, max_faces_per_bin=24000)
        renderer = MeshRendererWithFragments(rasterizer=MeshRasterizer(cameras=camera_train, raster_settings=raster_settings), shader=shader).to(device)
        _, fragments = renderer(new_mesh)
        uv_coords = get_uv_coordinates(new_mesh, fragments).cpu()
        uv_coords_eval_list.append(uv_coords)
        # depth tensor
        rel_depth = get_relative_depth_map(fragments.zbuf)
        rel_depth_normalized = rel_depth.unsqueeze(1).to(device)
        assert len(rel_depth_normalized.shape) == 4
        rel_depth_normalized = rel_depth_normalized.repeat(1, 3, 1, 1).float()
        rel_depth_normalized = F.interpolate(rel_depth_normalized, (render_size, render_size), mode="bilinear", align_corners=False)
        # expected range [0,1]
        rel_depth_normalized /= 255.0
        depth_tensor = rel_depth_normalized.to(dtype_half).cpu()
        depth_tensor_eval_list.append(depth_tensor)

    # generate train uv coordinates, depth tensors, and angle deviations
    uv_coords_list = []
    depth_tensor_list = []
    angle_deviation_list = []
    pbar = tqdm(range(num_views))
    fov = 60
    print("Generating UV coordinates, depth tensors, angle deviations for each view...")
    for i in pbar:
        elev = random.uniform(0, 60)
        azim = random.uniform(-170, 180)

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=at)
        camera_train = FoVPerspectiveCameras(R=R, T=T, device=device, fov=fov)
        shader = FlatTexelShader(cameras=camera_train, device=device, blend_params=BlendParams())
        raster_settings = RasterizationSettings(image_size=render_size, faces_per_pixel=faces_per_pixel, max_faces_per_bin=24000)
        renderer = MeshRendererWithFragments(rasterizer=MeshRasterizer(cameras=camera_train, raster_settings=raster_settings), shader=shader).to(device)
        _, fragments = renderer(new_mesh)

        # uv coords
        uv_coords = get_uv_coordinates(new_mesh, fragments).cpu()
        uv_coords_list.append(uv_coords)

        # depth tensor
        rel_depth = get_relative_depth_map(fragments.zbuf)
        rel_depth_normalized = rel_depth.unsqueeze(1).to(device)
        assert len(rel_depth_normalized.shape) == 4
        rel_depth_normalized = rel_depth_normalized.repeat(1, 3, 1, 1).float()
        rel_depth_normalized = F.interpolate(rel_depth_normalized, (render_size, render_size), mode="bilinear", align_corners=False)
        # expected range [0,1]
        rel_depth_normalized /= 255.0
        depth_tensor = rel_depth_normalized.to(dtype_half).cpu()
        depth_tensor_list.append(depth_tensor)

        # angle deviation
        # Convert angles from degrees to radians
        elev_ref_rad = np.radians(30)
        azim_ref_rad = np.radians(0)
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)
        # Calculate the angle deviation
        cos_theta = (np.cos(elev_ref_rad) * np.cos(elev_rad) * np.cos(azim_ref_rad - azim_rad) + np.sin(elev_ref_rad) * np.sin(elev_rad))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure the value is within the valid range for arccos
        angle_deviation_list.append(cos_theta)
    
    return uv_coords_list, uv_coords_eval_list, depth_tensor_list, depth_tensor_eval_list, angle_deviation_list


def data_prepare_scannetpp(dtype_half, cameras, num_views_eval, num_views, device, dist, at, render_size, faces_per_pixel, new_mesh):
    # generate train uv coordinates, depth tensors, and angle deviations
    uv_coords_list = []
    depth_tensor_list = []
    pbar = tqdm(range(len(cameras)))
    print("Generating UV coordinates, depth tensors for each view...")
    for i in pbar:
        camera_train = cameras[i]
        shader = FlatTexelShader(cameras=camera_train, device=device, blend_params=BlendParams())
        raster_settings = RasterizationSettings(image_size=render_size, faces_per_pixel=faces_per_pixel, max_faces_per_bin=30000)
        renderer = MeshRendererWithFragments(rasterizer=MeshRasterizer(cameras=camera_train, raster_settings=raster_settings), shader=shader).to(device)
        _, fragments = renderer(new_mesh)

        # uv coords
        uv_coords = get_uv_coordinates(new_mesh, fragments).cpu()
        uv_coords_list.append(uv_coords)

        # depth tensor
        rel_depth = get_relative_depth_map(fragments.zbuf)
        rel_depth_normalized = rel_depth.unsqueeze(1).to(device)
        assert len(rel_depth_normalized.shape) == 4
        rel_depth_normalized = rel_depth_normalized.repeat(1, 3, 1, 1).float()
        rel_depth_normalized = F.interpolate(rel_depth_normalized, (render_size, render_size), mode="bilinear", align_corners=False)
        # expected range [0,1]
        rel_depth_normalized /= 255.0
        depth_tensor = rel_depth_normalized.to(dtype_half).cpu()
        depth_tensor_list.append(depth_tensor)
    
    return uv_coords_list, depth_tensor_list


def data_prepare_scannetpp_video_render(cameras, device, render_size, faces_per_pixel, new_mesh):
    # generate train uv coordinates, depth tensors, and angle deviations
    uv_coords_list = []
    pbar = tqdm(range(len(cameras)))
    print("Generating UV coordinatesfor each view...")
    for i in pbar:
        camera_train = cameras[i]
        shader = FlatTexelShader(cameras=camera_train, device=device, blend_params=BlendParams())
        raster_settings = RasterizationSettings(image_size=render_size, faces_per_pixel=faces_per_pixel, max_faces_per_bin=300000)
        renderer = MeshRendererWithFragments(rasterizer=MeshRasterizer(cameras=camera_train, raster_settings=raster_settings), shader=shader).to(device)
        _, fragments = renderer(new_mesh)

        # uv coords
        uv_coords = get_uv_coordinates(new_mesh, fragments).cpu()
        uv_coords_list.append(uv_coords)
    
    return uv_coords_list


def data_prepare_scannetpp_angle_translation_deviation(dtype_half, cameras, device, render_size, faces_per_pixel, new_mesh, secret_view_idx):
    # generate train uv coordinates, depth tensors, and angle deviations
    uv_coords_list = []
    depth_tensor_list = []
    angle_deviation_list = []

    camera_ref = cameras[secret_view_idx]
    
    # Extract reference camera's rotation matrix for converting to spherical coordinates
    R_ref = camera_ref.R.squeeze(0).cpu().numpy()
    T_ref = camera_ref.T.squeeze(0).cpu().numpy()
    
    # Convert reference camera to spherical coordinates
    # Camera position in world coordinates
    ref_pos = -R_ref.T @ T_ref
    
    # Calculate reference elevation and azimuth from camera position
    # Assuming the object is at origin (0, 0, 0)
    ref_dist = np.linalg.norm(ref_pos)
    ref_elev_rad = np.arcsin(ref_pos[1] / ref_dist)  # elevation from y-coordinate
    ref_azim_rad = np.arctan2(ref_pos[0], ref_pos[2])  # azimuth from x,z coordinates
    
    # Compute distance scale from all camera positions for normalization
    print("Computing distance scale from camera positions...")
    all_positions = []
    for cam in cameras:
        R = cam.R.squeeze(0).cpu().numpy()
        T = cam.T.squeeze(0).cpu().numpy()
        pos = -R.T @ T
        all_positions.append(pos)
    
    all_positions = np.array(all_positions)
    
    # Calculate pairwise distances between all cameras
    distances = []
    for i in range(len(all_positions)):
        for j in range(i + 1, len(all_positions)):
            dist = np.linalg.norm(all_positions[i] - all_positions[j])
            distances.append(dist)
    
    distances = np.array(distances)
    # Use mean distance as the scale parameter for position deviation
    distance_scale = np.mean(distances)
    print(f"Computed distance scale: {distance_scale:.3f}")
    print(f"Distance stats - min: {np.min(distances):.3f}, max: {np.max(distances):.3f}, std: {np.std(distances):.3f}")

    pbar = tqdm(range(len(cameras)))
    print("Generating UV coordinates, depth tensors, angle deviations for each view...")
    for i in pbar:
        camera_train = cameras[i]
        shader = FlatTexelShader(cameras=camera_train, device=device, blend_params=BlendParams())
        raster_settings = RasterizationSettings(image_size=render_size, faces_per_pixel=faces_per_pixel, max_faces_per_bin=300000)
        renderer = MeshRendererWithFragments(rasterizer=MeshRasterizer(cameras=camera_train, raster_settings=raster_settings), shader=shader).to(device)
        _, fragments = renderer(new_mesh)

        # uv coords
        uv_coords = get_uv_coordinates(new_mesh, fragments).cpu()
        uv_coords_list.append(uv_coords)

        # depth tensor
        rel_depth = get_relative_depth_map(fragments.zbuf)
        rel_depth_normalized = rel_depth.unsqueeze(1).to(device)
        assert len(rel_depth_normalized.shape) == 4
        rel_depth_normalized = rel_depth_normalized.repeat(1, 3, 1, 1).float()
        rel_depth_normalized = F.interpolate(rel_depth_normalized, (render_size, render_size), mode="bilinear", align_corners=False)
        # expected range [0,1]
        rel_depth_normalized /= 255.0
        depth_tensor = rel_depth_normalized.to(dtype_half).cpu()
        depth_tensor_list.append(depth_tensor)

        # Combined angle and position deviation calculation
        # Get current camera's rotation matrix and translation
        R_curr = camera_train.R.squeeze(0).cpu().numpy()
        T_curr = camera_train.T.squeeze(0).cpu().numpy()
        
        # Calculate current camera position in world coordinates
        curr_pos = -R_curr.T @ T_curr
        
        # 1. Angular deviation using spherical coordinates
        curr_dist = np.linalg.norm(curr_pos)
        curr_elev_rad = np.arcsin(curr_pos[1] / curr_dist)  # elevation from y-coordinate
        curr_azim_rad = np.arctan2(curr_pos[0], curr_pos[2])  # azimuth from x,z coordinates
        
        # Calculate the angle deviation using the spherical coordinate formula
        angular_similarity = (np.cos(ref_elev_rad) * np.cos(curr_elev_rad) * np.cos(ref_azim_rad - curr_azim_rad) + 
                             np.sin(ref_elev_rad) * np.sin(curr_elev_rad))
        
        # 2. Position deviation (translation similarity)
        pos_diff = curr_pos - ref_pos
        translation_distance = np.linalg.norm(pos_diff)
        
        # Convert translation distance to a similarity measure using computed scale
        # This gives values in [0, 1] where 1 means same position, ~0.37 at mean distance
        position_similarity = np.exp(-translation_distance / distance_scale)
        
        # 3. Combine angular and position similarities
        alpha = 0.7  # Weight for angular similarity (dominant)
        beta = 1 - alpha   # Weight for position similarity
        combined_similarity = alpha * angular_similarity + beta * position_similarity
        
        # Clip to valid cosine range [-1, 1]
        cos_theta = np.clip(combined_similarity, -1.0, 1.0)
        angle_deviation_list.append(cos_theta)

    return uv_coords_list, depth_tensor_list, angle_deviation_list


def data_prepare_scannetpp_ves(dtype_half, cameras, ves_cameras, device, render_size, faces_per_pixel, new_mesh):
    # generate train uv coordinates, depth tensors, and angle deviations
    uv_coords_list = []
    depth_tensor_list = []
    pbar = tqdm(range(len(cameras)))
    print("Generating UV coordinates, depth tensors for each view...")
    for i in pbar:
        camera_train = cameras[i]
        shader = FlatTexelShader(cameras=camera_train, device=device, blend_params=BlendParams())
        raster_settings = RasterizationSettings(image_size=render_size, faces_per_pixel=faces_per_pixel, max_faces_per_bin=30000)
        renderer = MeshRendererWithFragments(rasterizer=MeshRasterizer(cameras=camera_train, raster_settings=raster_settings), shader=shader).to(device)
        _, fragments = renderer(new_mesh)

        # uv coords
        uv_coords = get_uv_coordinates(new_mesh, fragments).cpu()
        uv_coords_list.append(uv_coords)

        # depth tensor
        rel_depth = get_relative_depth_map(fragments.zbuf)
        rel_depth_normalized = rel_depth.unsqueeze(1).to(device)
        assert len(rel_depth_normalized.shape) == 4
        rel_depth_normalized = rel_depth_normalized.repeat(1, 3, 1, 1).float()
        rel_depth_normalized = F.interpolate(rel_depth_normalized, (render_size, render_size), mode="bilinear", align_corners=False)
        # expected range [0,1]
        rel_depth_normalized /= 255.0
        depth_tensor = rel_depth_normalized.to(dtype_half).cpu()
        depth_tensor_list.append(depth_tensor)

    # generate ves uv coordinates, depth tensors
    uv_coords_ves_list = []
    depth_tensor_ves_list = []
    pbar_ves = tqdm(range(len(ves_cameras)))
    print("Generating VES UV coordinates, depth tensors for each view...")
    for i in pbar_ves:
        camera_train = ves_cameras[i]
        shader = FlatTexelShader(cameras=camera_train, device=device, blend_params=BlendParams())
        raster_settings = RasterizationSettings(image_size=render_size, faces_per_pixel=faces_per_pixel, max_faces_per_bin=30000)
        renderer = MeshRendererWithFragments(rasterizer=MeshRasterizer(cameras=camera_train, raster_settings=raster_settings), shader=shader).to(device)
        _, fragments = renderer(new_mesh)

        # uv coords
        uv_coords = get_uv_coordinates(new_mesh, fragments).cpu()
        uv_coords_ves_list.append(uv_coords)

        # depth tensor
        rel_depth = get_relative_depth_map(fragments.zbuf)
        rel_depth_normalized = rel_depth.unsqueeze(1).to(device)
        assert len(rel_depth_normalized.shape) == 4
        rel_depth_normalized = rel_depth_normalized.repeat(1, 3, 1, 1).float()
        rel_depth_normalized = F.interpolate(rel_depth_normalized, (render_size, render_size), mode="bilinear", align_corners=False)
        # expected range [0,1]
        rel_depth_normalized /= 255.0
        depth_tensor = rel_depth_normalized.to(dtype_half).cpu()
        depth_tensor_ves_list.append(depth_tensor)
    
    return uv_coords_list, depth_tensor_list, uv_coords_ves_list, depth_tensor_ves_list


def make_rotation_matrices(delta_theta, delta_phi):
    """
    Given two angles (in radians), return the 3×3 rotation matrices
    Rx(Δθ) (around X axis) and Ry(Δφ) (around Y axis), exactly as in the paper:

    Rx(Δθ) = [[1,       0,        0     ],
                [0,  cosΔθ,  -sinΔθ ],
                [0,  sinΔθ,   cosΔθ ]]

    Ry(Δφ) = [[ cosΔφ,  0,  sinΔφ ],
                [   0,        1,      0   ],
                [ -sinΔφ,  0,  cosΔφ ]]

    Returns:
    Rx (3×3 numpy), Ry (3×3 numpy)
    """
    ct = np.cos(delta_theta)
    st = np.sin(delta_theta)
    Rx = np.array([
        [1.0,  0.0,  0.0],
        [0.0,   ct,  -st],
        [0.0,   st,   ct]
    ], dtype=np.float32)

    cp = np.cos(delta_phi)
    sp = np.sin(delta_phi)
    Ry = np.array([
        [ cp,  0.0,  sp],
        [ 0.0, 1.0,  0.0],
        [-sp,  0.0,  cp]
    ], dtype=np.float32)

    return Rx, Ry


def generate_ves_poses(c2w_secret, angle_limit_degrees=15.0):
    """
    Given a single camera-to-world matrix (4×4) as `c2w_secret` (numpy array),
    produce a list of “VES” camera-to-world matrices by rotating ±angle_limit
    around X and Y. We follow exactly Algorithm 1 from the VES paper:

    For Δθ, Δφ in {−δ, 0, +δ}^2 \ {(0,0)}:
        R'_w2c = Rx(Δθ) @ Ry(Δφ) @ R_w2c_secret
        t_w2c remains the same
        then invert back to c2w

    Inputs:
    c2w_secret : numpy.ndarray of shape (4, 4)
        - The “secret” camera‐to‐world matrix, e.g. train_c2w[secret_view_idx], in OpenCV convention.
    angle_limit_degrees : float
        - The maximum positive/negative rotation (in degrees) to apply around X and Y.

    Returns:
    ves_c2w_list : list of numpy.ndarray of shape (4, 4)
        - A list containing 8 new camera‐to‐world matrices, each rotated by ±δ along X/Y.
    """
    # 1) Convert the secret c2w into w2c (i.e. R_w2c, t_w2c).
    #    If c2w_secret = [ R_c2w | t_c2w ]
    #                       [   0   |    1    ],
    #    then R_w2c = R_c2w^T,  t_w2c = - R_c2w^T @ t_c2w.
    R_c2w = c2w_secret[:3, :3]
    t_c2w = c2w_secret[:3, 3 : 4]  # shape (3,1)

    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w  # shape (3,1)

    # 2) Build all combinations of Δθ and Δφ in {−δ, 0, +δ}, except (0,0).
    δ_rad = np.deg2rad(angle_limit_degrees)
    deltas = [ -δ_rad, 0.0, +δ_rad ]

    ves_c2w_list = []
    for dtheta in deltas:
        for dphi in deltas:
            # Skip the (0,0) case because that’s just the original pose.
            if np.isclose(dtheta, 0.0) and np.isclose(dphi, 0.0):
                continue

            # 3) Compute Rx and Ry for these small angles:
            Rx, Ry = make_rotation_matrices(dtheta, dphi)

            # 4) Rotate the original world‐to‐camera rotation:
            #    R'_w2c = Rx @ Ry @ R_w2c
            Rprime_w2c = Rx @ (Ry @ R_w2c)

            # 5) t_w2c stays the same, so we have [R'_w2c | t_w2c].
            #    Now convert back to camera-to-world:
            #    R'_c2w = (R'_w2c)^T
            #    t'_c2w = - R'_c2w @ (t_w2c)
            Rprime_c2w = Rprime_w2c.T
            tprime_c2w = -Rprime_c2w @ t_w2c

            # 6) Assemble the new 4×4 c2w matrix:
            c2w_prime = np.eye(4, dtype=np.float32)
            c2w_prime[:3, :3] = Rprime_c2w
            c2w_prime[:3, 3 : 4] = tprime_c2w

            ves_c2w_list.append(c2w_prime)

    return ves_c2w_list


def query_texture(uv_coords, model, device):
    B, H, W, C = uv_coords.shape
    inputs = uv_coords.reshape(-1, C)
    outputs = model(inputs)
    rgb = outputs.reshape(B, H, W, -1).to(device)

    return rgb


def query_texture_sh(uv_coords, rays, model, device):
    outputs = model(uv_coords, rays).to(device)

    return outputs


def get_rgb(uv_coords, zero_mask, model, device):
    rgb = query_texture(uv_coords, model, device)

    expanded_zero_mask = zero_mask.unsqueeze(-1).expand_as(rgb)
    rgb = rgb.masked_fill(expanded_zero_mask, -1)

    rgb = rgb.permute(0, 3, 1, 2).to(device)
    rgb = rgb.clamp(-1, 1)

    return rgb


def save_img_results(num_views_eval, particles, device, rgb_path, step, uv_coords_eval_list, eval_rgb):
    rgb_list = []
    eval_rgb_list = []
    for view_eval_id in range(num_views_eval):
        for model in particles:
            uv_coords = uv_coords_eval_list[view_eval_id].to(device)
            zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
            rgb = query_texture(uv_coords, model, device)
            expanded_zero_mask = zero_mask.unsqueeze(-1).expand_as(rgb)
            rgb = rgb.masked_fill(expanded_zero_mask, 1)
            rgb = rgb.permute(0, 3, 1, 2).to(device)
            rgb = rgb.clamp(-1, 1)
        # save rgb results
        rgb_img = rgb.clone().detach().cpu().squeeze(0)
        rgb_list.append(rgb_img)
        eval_img = ((eval_rgb[view_eval_id] - 0.5) * 2).clone().permute(2, 0, 1).detach().cpu()
        eval_rgb_list.append(eval_img)

    rgb_imgs = torch.stack(rgb_list, dim=0)
    eval_imgs = torch.stack(eval_rgb_list, dim=0)

    img = torch.cat([rgb_imgs, eval_imgs], dim=2)

    save_image((img / 2 + 0.5).clamp(0, 1), rgb_path + f'/rgb_{step}_predict.png')


def save_img_results_sh(num_views_eval, particles, device, rgb_path, step, uv_coords_eval_list, rays_eval_list, eval_rgb):
    rgb_list = []
    eval_rgb_list = []
    for view_eval_id in range(num_views_eval):
        for model in particles:
            uv_coords = uv_coords_eval_list[view_eval_id].to(device)
            zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
            rays = rays_eval_list[view_eval_id].to(device)
            rgb = query_texture_sh(uv_coords, rays, model, device)
            expanded_zero_mask = zero_mask.unsqueeze(-1).expand_as(rgb)
            rgb = rgb.masked_fill(expanded_zero_mask, 1)
            rgb = rgb.permute(0, 3, 1, 2).to(device)
            rgb = rgb.clamp(-1, 1)
        # save rgb results
        rgb_img = rgb.clone().detach().cpu().squeeze(0)
        rgb_list.append(rgb_img)
        eval_img = ((eval_rgb[view_eval_id] - 0.5) * 2).clone().permute(2, 0, 1).detach().cpu()
        eval_rgb_list.append(eval_img)

    rgb_imgs = torch.stack(rgb_list, dim=0)
    eval_imgs = torch.stack(eval_rgb_list, dim=0)

    img = torch.cat([rgb_imgs, eval_imgs], dim=2)

    save_image((img / 2 + 0.5).clamp(0, 1), rgb_path + f'/rgb_{step}_predict.png')


def save_eval_results(num_views_eval, particles, device, rgb_path, step, uv_coords_eval_list):
    rgb_list = []
    for view_eval_id in range(num_views_eval):
        for model in particles:
            uv_coords = uv_coords_eval_list[view_eval_id].to(device)
            zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
            rgb = query_texture(uv_coords, model, device)
            expanded_zero_mask = zero_mask.unsqueeze(-1).expand_as(rgb)
            rgb = rgb.masked_fill(expanded_zero_mask, 1)
            rgb = rgb.permute(0, 3, 1, 2).to(device)
            rgb = rgb.clamp(-1, 1).float()
        # save rgb results
        rgb_img = rgb.clone().detach().cpu().squeeze(0)
        rgb_list.append(rgb_img)

    rgb_imgs = torch.stack(rgb_list, dim=0)

    save_image((rgb_imgs / 2 + 0.5).clamp(0, 1), rgb_path + f'/eval_{step}_predict.png')


def inference(texture_size, particles, device, rgb_path, step):
    with torch.no_grad():
        u, v = torch.arange(texture_size).to(device), torch.arange(texture_size).to(device)
        u, v = torch.meshgrid(u, v, indexing='ij')
        inputs_texture = torch.stack([u, v]).permute(1, 2, 0).unsqueeze(0) / (texture_size - 1)

        texture = torch.zeros(texture_size, texture_size, 3, device=device)

        for row in tqdm(range(texture_size)):
            for model in particles:
                r_inputs = inputs_texture[:, row].float()
                r_inputs = r_inputs.unsqueeze(1)

                rgb = query_texture(r_inputs, model, device)

                texture[row] = rgb.detach()

        texture = (texture / 2 + 0.5).clamp(0, 1)
        texture = texture.cpu()
        # Permute from (H, W, C) to (W, H, C)
        texture = texture.permute(1, 0, 2)
        # Flip vertically
        texture = torch.flip(texture, dims=[0])
        assert texture.min() >= 0 and texture.max() <= 1

        texture = torchvision.transforms.ToPILImage()(texture.permute(2, 0, 1)).convert("RGB")
        texture.save(os.path.join(rgb_path, f"texture_{step}.png"))


def hirarchical_inference(texture_size, particles, device, rgb_path, step):
    with torch.no_grad():
        spacing = 10
        for model in particles:
            grid_textures = []
            for i, l in enumerate(model.layers):
                u, v = torch.arange(texture_size).to(device), torch.arange(texture_size).to(device)
                u, v = torch.meshgrid(u, v, indexing='ij')
                inputs_texture = torch.stack([u, v]).permute(1, 2, 0).unsqueeze(0) / (texture_size - 1)

                texture = torch.zeros(texture_size, texture_size, 3, device=device)

                for row in tqdm(range(texture_size)):
                    r_inputs = inputs_texture[:, row].float()
                    r_inputs = r_inputs.unsqueeze(1)

                    rgb = query_texture(r_inputs, l, device)

                    texture[row] = rgb.detach()

                texture = (texture / 2 + 0.5).clamp(0, 1)
                texture = texture.cpu()
                # Permute from (H, W, C) to (W, H, C)
                texture = texture.permute(1, 0, 2)
                # Flip vertically
                texture = torch.flip(texture, dims=[0])
                assert texture.min() >= 0 and texture.max() <= 1

                texture = torchvision.transforms.ToPILImage()(texture.permute(2, 0, 1)).convert("RGB")
                grid_textures.append(texture)
                # texture.save(os.path.join(rgb_path, f"texture_{step}_layer_{i}.png"))

            # Ensure there are exactly four layers
            assert len(grid_textures) == 4, f"Model must have exactly 4 layers for a 2x2 grid"

            # Create a 2x2 grid using Matplotlib
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            for ax, texture in zip(axs.flat, grid_textures):
                ax.imshow(texture)
                ax.axis('off')  # Turn off axis labels and ticks
            plt.tight_layout()

            # Save the grid image
            grid_filename = os.path.join(rgb_path, f"texture_{step}_layers.png")

            plt.savefig(grid_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory


def save_scannetpp_results(num_views_eval, images, uv_coords_list, model, device, i, rgb_path_stage_2):
    rgb_list = []
    eval_rgb_list = []
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

    save_image((img / 2 + 0.5).clamp(0, 1), rgb_path_stage_2 + f'/eval_{i}_predict.png')


def save_ves_results(rgb_secret, uv_coords_ves_list, model, device, i, rgb_path_stage_2):
    rgb_list = []
    for view_eval_id in range(8):
        uv_coords = uv_coords_ves_list[view_eval_id].to(device)
        zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
        rgb = get_rgb(uv_coords, zero_mask, model, device)

        rgb_img = rgb.clone().detach().cpu().squeeze(0)
        rgb_list.append(rgb_img)

    # Make sure `rgb_secret` is also [3, H, W]
    if rgb_secret.dim() == 4:
        secret_img = rgb_secret.clone().detach().cpu().squeeze(0)  # [3, H, W]
    else:
        secret_img = rgb_secret.clone().detach().cpu()            # already [3, H, W]

    # Now arrange them into rows of three:
    # ┌–––––––––┬––––––––─┬––––––––––─┐
    # │ rgb_list[5] │ rgb_list[6] │ rgb_list[7] │  ← row1
    # ├––––––––─┼––––––––─┼––––––––––─┤
    # │ rgb_list[3] │ secret_img  │ rgb_list[4] │  ← row2 (secret in center)
    # ├––––––––─┼––––––––─┼––––––––––─┤
    # │ rgb_list[0] │ rgb_list[1] │ rgb_list[2] │  ← row3
    # └––––––––─┴––––––––─┴––––––––––─┘

    row1 = torch.cat([rgb_list[5], rgb_list[6],     rgb_list[7]], dim=2)
    row2 = torch.cat([rgb_list[3], secret_img,      rgb_list[4]], dim=2)
    row3 = torch.cat([rgb_list[0], rgb_list[1], rgb_list[2]], dim=2)  # concat along W

    # Now stack the three rows along H to get a single [3, 3H, 3W] image
    img = torch.cat([row1, row2, row3], dim=1)  # concat along H

    save_image((img / 2 + 0.5).clamp(0, 1), rgb_path_stage_2 + f'/ves_results_{i}_predict.png')


# Show a visualization comparing the rendered predicted mesh to the ground truth 
# mesh
def visualize_prediction(predicted_image, target_image, save_path, title='', silhouette=False):
    inds = range(3)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_image[0, ..., inds].cpu().detach().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")

    # Save the figure to the specified path instead of showing it
    plt.savefig(save_path)
    plt.close()


# Plot losses as a function of optimization iteration
def plot_losses(losses, save_path):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l['values'], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")

    # Save the figure to the specified path instead of showing it
    plt.savefig(save_path)
    plt.close()


# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)
    
    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)
    
    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")