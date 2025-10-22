# https://pytorch3d.org/tutorials/fit_textured_mesh
# Tutorial: Fit a Textured Mesh from Images

######################################################
# Import modules                                     
######################################################
import os
import torch
from torch import nn
from torchvision.utils import save_image

import numpy as np
from tqdm import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

# utils
from utils.pytorch3d_uv_utils import (
    dataset_prepare,
    generate_uv_coords_from_mesh,
    plot_losses,
    generate_uv_mapping_mesh,
    query_texture,
    inference
)

# Neural texture
from model.neural_texture import RGB_field


# main function
def fit_texture_via_rendering():
    ######################################################
    # Load a mesh and set configs                                  
    ######################################################
    # 1. Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Texture
    latent_texture_size = 256
    latent_channels = 3
    texture_size = 2048

    # Set paths
    DATA_DIR = "./data"
    obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

    output_path = './outputs/pytorch3d_uv'
    os.makedirs(output_path, exist_ok=True)

    rgb_dataset_train_path = output_path + '/rgb_train_uv'
    os.makedirs(rgb_dataset_train_path, exist_ok=True)

    rgb_path = output_path + '/rgb_uv'
    os.makedirs(rgb_path, exist_ok=True)

    # the number of different viewpoints from which we want to render the mesh.
    num_views = 100
    render_size = 768
    faces_per_pixel = 1
    dist = 2.7

    # Number of particles in the VSD
    particle_num_vsd = 1

    # Number of views to optimize over in each SGD iteration
    num_views_per_iteration = 1
    # Number of optimization steps
    Niter = 5000
    # Plot period for the losses
    plot_period = 1000
    # Learning rate
    lr = 1e-2
    eps = 1e-9
    weight_decay = 0.1
    # good settings
    # lr = 1e-2
    # eps = 1e-9
    # weight_decay = 0.1


    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 
    # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh 
    # to its original center and scale.  Note that normalizing the target mesh, 
    # speeds up the optimization but is not necessary!
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))

    # get or generate uv parameterization
    new_mesh = generate_uv_mapping_mesh(obj_filename, output_path, device, latent_texture_size, latent_channels)

    ######################################################
    # Dataset and learnable model creation                                     
    ######################################################
    # 2. Setup the renderer
    # Prepare the dataset
    target_rgb = dataset_prepare(num_views, device, dist, render_size, faces_per_pixel, mesh, rgb_dataset_train_path)

    # Prepare the uv coordinates for each view
    uv_coords_list = generate_uv_coords_from_mesh(num_views, render_size, faces_per_pixel, dist, device, new_mesh)

    particle_models = [
        RGB_field(texture_size).to(device)
        for _ in range(particle_num_vsd)
    ]

    particles = nn.ModuleList(particle_models)

    particles_to_optimize = [param for hashmlp in particles for param in hashmlp.parameters() if param.requires_grad]
    texture_params = [p for p in particles_to_optimize if p.requires_grad]
    print("=> Total number of trainable parameters for texture: {}".format(sum(p.numel() for p in texture_params if p.requires_grad)))

    ######################################################
    # Texture prediction via uv texture mapping                                    
    ######################################################
    losses = {"rgb": {"weight": 1.0, "values": []}}

    # The optimizer
    optimizer = torch.optim.AdamW(particles_to_optimize, lr=lr, eps=eps, weight_decay=weight_decay)

    loop = tqdm(range(Niter))

    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        
        # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=device) for k in losses}
        
        # Randomly select two views to optimize over in this iteration.  Compared
        # to using just one view, this helps resolve ambiguities between updating
        # mesh shape vs. updating mesh texture
        for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
            for model in particles:
                uv_coords = uv_coords_list[j].to(device)
                zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)

                rgb = query_texture(uv_coords, model, device)

                expanded_zero_mask = zero_mask.unsqueeze(-1).expand_as(rgb)
                rgb = rgb.masked_fill(expanded_zero_mask, 1)

                rgb = rgb.permute(0, 3, 1, 2).to(device)
                rgb = rgb.clamp(-1, 1)
                
                # Squared L2 distance between the predicted RGB image and the target 
                # image from our dataset
                predicted_rgb = rgb.squeeze(0).permute(1, 2, 0)
                loss_rgb_1 = ((predicted_rgb - ((target_rgb[j] - 0.5) * 2)) ** 2).mean()

                uv_coords_ = uv_coords_list[0].to(device)
                zero_mask_ = (uv_coords_[..., 0] == 0) & (uv_coords_[..., 1] == 0)

                rgb_ = query_texture(uv_coords_, model, device)

                expanded_zero_mask_ = zero_mask_.unsqueeze(-1).expand_as(rgb_)
                rgb_ = rgb_.masked_fill(expanded_zero_mask_, 1)

                rgb_ = rgb_.permute(0, 3, 1, 2).to(device)
                rgb_ = rgb_.clamp(-1, 1)
                
                # Squared L2 distance between the predicted RGB image and the target 
                # image from our dataset
                predicted_rgb_ = rgb_.squeeze(0).permute(1, 2, 0)
                loss_rgb_2 = ((predicted_rgb_ - ((target_rgb[0] - 0.5) * 2)) ** 2).mean()

                loss["rgb"] += (loss_rgb_1 + loss_rgb_2) / (num_views_per_iteration * 2)
        
        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=device)
        for k, l in loss.items():
            sum_loss += l * losses[k]["weight"]
            losses[k]["values"].append(float(l.detach().cpu()))
        
        # Print the losses
        loop.set_description("total_loss = %.6f" % sum_loss)
        
        # Plot mesh
        if i % plot_period == 0:
            # save rgb results
            rgb_img = rgb.clone().detach().cpu().squeeze(0)
            target_img = ((target_rgb[j] - 0.5) * 2).clone().permute(2, 0, 1).detach().cpu()
            rgb_img_secret = rgb_.clone().detach().cpu().squeeze(0)
            target_img_secret = ((target_rgb[0] - 0.5) * 2).clone().permute(2, 0, 1).detach().cpu()
            img = torch.cat([rgb_img, target_img, rgb_img_secret, target_img_secret], dim=2)
            save_image((img / 2 + 0.5).clamp(0, 1), rgb_path + f'/rgb_{i}_predict.png')
            # inference
            inference(texture_size, particles, device, rgb_path, i)

        # Optimization step
        sum_loss.backward()
        optimizer.step()
    
    rgb_img = rgb.clone().detach().cpu().squeeze(0)
    target_img = ((target_rgb[j] - 0.5) * 2).clone().permute(2, 0, 1).detach().cpu()
    rgb_img_secret = rgb_.clone().detach().cpu().squeeze(0)
    target_img_secret = ((target_rgb[0] - 0.5) * 2).clone().permute(2, 0, 1).detach().cpu()
    img = torch.cat([rgb_img, target_img, rgb_img_secret, target_img_secret], dim=2)
    save_image((img / 2 + 0.5).clamp(0, 1), rgb_path + f'/rgb_final_predict.png')
    # inference
    inference(texture_size, particles, device, rgb_path, i + 1)

    # plot losses
    loss_path = rgb_path + f'/rgb_loss.png'
    plot_losses(losses, loss_path)