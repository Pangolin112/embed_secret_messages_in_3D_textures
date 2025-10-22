# https://pytorch3d.org/tutorials/fit_textured_mesh
# Tutorial: Fit a Textured Mesh from Images

######################################################
# Import modules                                     
######################################################
import os
import torch
import matplotlib.pyplot as plt

from pytorch3d.utils import ico_sphere
import numpy as np
from tqdm import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

from utils.plot_image_grid import image_grid


######################################################
# Custom functions                                     
######################################################
# Show a visualization comparing the rendered predicted mesh to the ground truth 
# mesh
def visualize_prediction(predicted_mesh, renderer, target_image, save_path, title='', silhouette=False):
    inds = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_images = renderer(predicted_mesh)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())

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


# main function
def fit_mesh_via_rendering():
    ######################################################
    # Load a mesh and texture file                                     
    ######################################################
    # 1. Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Set paths
    DATA_DIR = "./data"
    obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 
    # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh 
    # to its original center and scale.  Note that normalizing the target mesh, 
    # speeds up the optimization but is not necessary!
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))


    ######################################################
    # Dataset creation                                     
    ######################################################
    # 2. Setup the renderer
    # the number of different viewpoints from which we want to render the mesh.
    num_views = 30
    render_size = 512
    faces_per_pixel = 50
    dist = 2.7

    # paths
    rgb_dataset_train_path = './outputs/pytorch3d/rgb_train'
    os.makedirs(rgb_dataset_train_path, exist_ok=True)
    silhouette_dataset_train_path = './outputs/pytorch3d/silhouette_train'
    os.makedirs(silhouette_dataset_train_path, exist_ok=True)

    output_path = './outputs/pytorch3d'
    silhouette_path = output_path + '/silhouette'
    rgb_path = output_path + '/rgb'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(silhouette_path, exist_ok=True)
    os.makedirs(rgb_path, exist_ok=True)

    # Get a batch of viewing angles. 
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)

    # Place a point light in front of the object. As mentioned above, the front of 
    # the cow is facing the -z direction. 
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Initialize an OpenGL perspective camera that represents a batch of different 
    # viewing angles. All the cameras helper methods support mixed type inputs and 
    # broadcasting. So we can view the camera from a distance of dist=2.7, and 
    # then specify elevation and azimuth angles for each viewpoint as tensors. 
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # We arbitrarily choose one particular view that will be used to visualize 
    # results
    # camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...], T=T[None, 1, ...]) 
                                    
    elev_eval = (elev[None, 1, ...] + elev[None, 2, ...]) / 2
    azim_eval = (azim[None, 1, ...] + azim[None, 2, ...]) / 2
    R_eval, T_eval = look_at_view_transform(dist=dist, elev=elev_eval, azim=azim_eval)
    camera = FoVPerspectiveCameras(device=device, R=R_eval, T=T_eval) 
                                    

    # Define the settings for rasterization and shading. Here we set the output 
    # image to be of size 128X128. As we are rendering images for visualization 
    # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to 
    # rasterize_meshes.py for explanations of these parameters.  We also leave 
    # bin_size and max_faces_per_bin to their default values of None, which sets 
    # their values using heuristics and ensures that the faster coarse-to-fine 
    # rasterization method is used.  Refer to docs/notes/renderer.md for an 
    # explanation of the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=render_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured 
    # Phong shader will interpolate the texture uv coordinates for each vertex, 
    # sample from a texture image and apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=camera,
            lights=lights
        )
    )

    # Create a batch of meshes by repeating the cow mesh and associated textures. 
    # Meshes has a useful `extend` method which allows us do this very easily. 
    # This also extends the textures. 
    meshes = mesh.extend(num_views)

    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras, lights=lights)

    # Our multi-view cow dataset will be represented by these 2 lists of tensors,
    # each of length num_views.
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)] # :3 omits the alpha channel
    target_cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...], 
                                            T=T[None, i, ...]) for i in range(num_views)]
    
    # RGB images
    image_grid(target_images.cpu().numpy(), rows=5, cols=6, rgb=True)
    # plt.show()
    # Save the figure to the specified path instead of showing it
    save_path = rgb_dataset_train_path + '/rgb_dataset_train.png'
    plt.savefig(save_path)
    plt.close()

    # 3. Setup the rasterization
    # Rasterization settings for silhouette rendering  
    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(
        image_size=render_size, 
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=faces_per_pixel, 
    )

    # Silhouette renderer 
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader()
    )

    # Render silhouette images.  The 3rd channel of the rendering output is 
    # the alpha/silhouette channel
    silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)
    target_silhouette = [silhouette_images[i, ..., 3] for i in range(num_views)] # 3 is the alpha/silhouette channel

    # Visualize silhouette images
    image_grid(silhouette_images.cpu().numpy(), rows=5, cols=6, rgb=False)
    # plt.show()
    save_path = silhouette_dataset_train_path + '/silhouette_dataset_train.png'
    plt.savefig(save_path)
    plt.close()


    ######################################################
    # Mesh prediction via silhouette rendering                                     
    ######################################################
    # Learn offsets of each vertex such that the predicted mesh silhouette aligns the target silhouette image.

    # We initialize the source shape to be a sphere of radius 1.  
    src_mesh = ico_sphere(4, device)

    # 4. Create a new differentiable renderer for rendering the silhouette of predicted mesh
    # Rasterization settings for differentiable rendering, where the blur_radius
    # initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable 
    # Renderer for Image-based 3D Reasoning', ICCV 2019
    sigma = 1e-4
    raster_settings_soft = RasterizationSettings(
        image_size=render_size, 
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=faces_per_pixel, 
    )

    # Silhouette renderer 
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_soft
        ),
        shader=SoftSilhouetteShader()
    )

    # 5. Initialize the training settings
    # Number of views to optimize over in each SGD iteration
    num_views_per_iteration = 2
    # Number of optimization steps
    Niter = 2000
    # Plot period for the losses
    plot_period = 250

    # Optimize using rendered silhouette image loss, mesh edge loss, mesh normal 
    # consistency, and mesh laplacian smoothing
    losses = {"silhouette": {"weight": 1.0, "values": []},
            "edge": {"weight": 1.0, "values": []},
            "normal": {"weight": 0.01, "values": []},
            "laplacian": {"weight": 1.0, "values": []},
            }

    # We will learn to deform the source mesh by offsetting its vertices
    # The shape of the deform parameters is equal to the total number of vertices in
    # src_mesh
    verts_shape = src_mesh.verts_packed().shape
    deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)

    # The optimizer
    optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

    # 6. Training loop
    loop = tqdm(range(Niter))

    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        
        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        
        # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=device) for k in losses}
        update_mesh_shape_prior_losses(new_src_mesh, loss)
        
        # Compute the average silhouette loss over two random views, as the average 
        # squared L2 distance between the predicted silhouette and the target 
        # silhouette from our dataset
        for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
            images_predicted = renderer_silhouette(new_src_mesh, cameras=target_cameras[j], lights=lights)
            predicted_silhouette = images_predicted[..., 3]
            loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
            loss["silhouette"] += loss_silhouette / num_views_per_iteration
        
        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=device)
        for k, l in loss.items():
            sum_loss += l * losses[k]["weight"]
            losses[k]["values"].append(float(l.detach().cpu()))

        # Print the losses
        loop.set_description("total_loss = %.6f" % sum_loss)
        
        # Plot mesh
        if i % plot_period == 0:
            save_path = silhouette_path + f'/silhouette_{i}.png'
            visualize_prediction(new_src_mesh, renderer_silhouette, target_silhouette[1], save_path, title="iter: %d" % i, silhouette=True)
            
        # Optimization step
        sum_loss.backward()
        optimizer.step()

    save_path = silhouette_path + f'/silhouette_final.png'
    loss_path = silhouette_path + f'/silhouette_loss.png'
    visualize_prediction(new_src_mesh, renderer_silhouette, target_silhouette[1], save_path, silhouette=True)
    plot_losses(losses, loss_path)

    
    ######################################################
    # Mesh and texture prediction via textured rendering                                     
    ######################################################
    # Learn both translational offsets and RGB texture colors for each vertex in the sphere mesh

    # Rasterization settings for differentiable rendering, where the blur_radius
    # initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable 
    # Renderer for Image-based 3D Reasoning', ICCV 2019
    sigma = 1e-4
    raster_settings_soft = RasterizationSettings(
        image_size=render_size, 
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=faces_per_pixel, 
        perspective_correct=False, 
    )

    # Differentiable soft renderer using per vertex RGB colors for texture
    renderer_textured = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_soft
        ),
        shader=SoftPhongShader(device=device, 
            cameras=camera,
            lights=lights)
    )

    # Number of views to optimize over in each SGD iteration
    num_views_per_iteration = 2
    # Number of optimization steps
    Niter = 2000
    # Plot period for the losses
    plot_period = 250

    # Optimize using rendered RGB image loss, rendered silhouette image loss, mesh 
    # edge loss, mesh normal consistency, and mesh laplacian smoothing
    losses = {"rgb": {"weight": 1.0, "values": []},
            "silhouette": {"weight": 1.0, "values": []},
            "edge": {"weight": 1.0, "values": []},
            "normal": {"weight": 0.01, "values": []},
            "laplacian": {"weight": 1.0, "values": []},
            }
    # losses = {"rgb": {"weight": 1.0, "values": []},
    #         "silhouette": {"weight": 0.0, "values": []},
    #         "edge": {"weight": 0.0, "values": []},
    #         "normal": {"weight": 0.0, "values": []},
    #         "laplacian": {"weight": 0.0, "values": []},
    #         }

    # We will learn to deform the source mesh by offsetting its vertices
    # The shape of the deform parameters is equal to the total number of vertices in 
    # src_mesh
    # verts_shape = src_mesh.verts_packed().shape
    # deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)

    # We will also learn per vertex colors for our sphere mesh that define texture 
    # of the mesh
    sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)

    # The optimizer
    optimizer = torch.optim.SGD([deform_verts, sphere_verts_rgb], lr=1.0, momentum=0.9)

    loop = tqdm(range(Niter))

    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        
        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        
        # Add per vertex colors to texture the mesh
        new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb) 
        
        # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=device) for k in losses}
        update_mesh_shape_prior_losses(new_src_mesh, loss)
        
        # Randomly select two views to optimize over in this iteration.  Compared
        # to using just one view, this helps resolve ambiguities between updating
        # mesh shape vs. updating mesh texture
        for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
            images_predicted = renderer_textured(new_src_mesh, cameras=target_cameras[j], lights=lights)

            # Squared L2 distance between the predicted silhouette and the target 
            # silhouette from our dataset
            predicted_silhouette = images_predicted[..., 3]
            loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
            loss["silhouette"] += loss_silhouette / num_views_per_iteration
            
            # Squared L2 distance between the predicted RGB image and the target 
            # image from our dataset
            predicted_rgb = images_predicted[..., :3]
            loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
            loss["rgb"] += loss_rgb / num_views_per_iteration
        
        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=device)
        for k, l in loss.items():
            sum_loss += l * losses[k]["weight"]
            losses[k]["values"].append(float(l.detach().cpu()))
        
        # Print the losses
        loop.set_description("total_loss = %.6f" % sum_loss)
        
        # Plot mesh
        if i % plot_period == 0:
            save_path = rgb_path + f'/rgb_{i}.png'
            visualize_prediction(new_src_mesh, renderer_textured, target_rgb[1], save_path, title="iter: %d" % i, silhouette=False)
            
        # Optimization step
        sum_loss.backward()
        optimizer.step()

    save_path = rgb_path + f'/rgb_final.png'
    loss_path = rgb_path + f'/rgb_loss.png'
    visualize_prediction(new_src_mesh, renderer_textured, target_rgb[1], save_path, silhouette=False)
    plot_losses(losses, loss_path)

    # Fetch the verts and faces of the final predicted mesh
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

    # Scale normalize back to the original target size
    final_verts = final_verts * scale + center

    # Store the predicted mesh using save_obj
    final_obj = os.path.join(output_path, 'final_model.obj')
    save_obj(final_obj, final_verts, final_faces)