######################################################
# Import modules and utils                                    
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

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# diffusers for loading sd model
from diffusers import DDIMScheduler, ControlNetModel, StableDiffusionControlNetPipeline

# add path for demo utils functions 
import os
import shutil
import yaml
from datetime import datetime

# utils
from utils.pytorch3d_uv_utils import (
    data_prepare,
    data_prepare_scannetpp,
    plot_losses,
    generate_uv_mapping_mesh,
    import_smart_uv_mesh,
    import_smart_uv_mesh_scannetpp,
    get_rgb,
    save_eval_results,
    inference,
    hirarchical_inference
)

from utils.sds_utils import (
    extract_lora_diffusers,
    encode_prompt_with_a_prompt_and_n_prompt,
    get_loss_weights,
    get_t_schedule_dreamtime,
    get_noisy_latents,
    get_noise_pred,
    phi_vsd_grad_diffuser
)

from utils.PTDiffusion_utils import (
    load_ref_img,
    encode_ddim,
    p_sample_ddim,
    p_sample_ddim_depth,
    phase_substitute
)

from utils.scannetpp_dataloader import (
    convert_nerfstudio_to_opencv,
    load_image,
    load_depth,
    rescale,
    rescale_depth,
    adjust_intrinsics,
    center_crop,
    rescale_and_crop,
    opencv_to_pt3d_cams
)

# Neural texture
from model.neural_texture import HierarchicalRGB_field
from model.hash_mlp import Hashgrid_MLP


# main function
def PTDiffusion_3d_RGB_optimize_wo_VSD():
    ######################################################
    # 1. Set configs                              
    ######################################################
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    config_path = 'config/config_PTDiffusion_3d.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

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
   
    # Set paths
    DATA_DIR = "./data"
    ply_filename = os.path.join(DATA_DIR, "ScanNetpp/meshes/49a82360aa/mesh_uv.ply")
    img_ref_name = 'face1.jpg'
    img_ref_path = os.path.join(DATA_DIR, "reference_images/" + img_ref_name)
    img_ref_save_name = img_ref_name.replace('.', '_')

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
        images = [load_image(os.path.join(rgb_root, frame["file_path"])) for frame in train_frames]
        images = torch.stack(images)  # (N, 3, h, w)

        # depth_root = os.path.join(args.input_dir, "data", scene_id, "dslr", "undistorted_render_depth")
        depth_root = os.path.join(ScanNetpp_path, "data", scene_id, "dslr", "render_depth")
        depth = [load_depth(os.path.join(depth_root, frame["file_path"].replace(".JPG", ".png"))) for frame in train_frames]
        depth = torch.stack(depth)  # (N, h, w)

        # center crop image, depth, intrinsics to (1024, 1024)
        # w = h = 1024
        w = h = render_size
        images, K, depth = rescale_and_crop(images, K, (h, w), depth)

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
    output_path = './outputs/PTDiffusion_3d_RGB_optimize_wo_VSD'
    os.makedirs(output_path, exist_ok=True)

    work_dir = output_path + '/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + f"_init_{init_flag}_cs_{conditioning_scale}_br_{blending_ratio_default}_tr_{transfer_ratio}_direct_{direct_transfer_ratio}_decayed_{decayed_transfer_ratio}_stage1_{Niter_stage_1}_stage2_{Niter_stage_2}_seed_{seed}_{prompt_1}_{prompt_2}"
    os.makedirs(work_dir, exist_ok=True)
    # save current file and config file to work_dir
    shutil.copyfile(__file__, os.path.join(work_dir, os.path.basename(__file__)))
    shutil.copy(config_path, os.path.join(work_dir, os.path.basename(config_path)))

    rgb_path_stage_1 = work_dir + '/rgb_uv_stage_1'
    os.makedirs(rgb_path_stage_1, exist_ok=True)
    rgb_path_stage_2 = work_dir + '/rgb_uv_stage_2'
    os.makedirs(rgb_path_stage_2, exist_ok=True)

    ######################################################
    # 2. Stable diffusion model loading                                     
    ######################################################
    dtype_half = torch.float16
    model_path = "runwayml/stable-diffusion-v1-5"
    controlnet_name = "lllyasviel/control_v11f1p_sd15_depth"
    controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=dtype_half, low_cpu_mem_usage=True).to(device)
    diffusion_model = StableDiffusionControlNetPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=dtype_half, low_cpu_mem_usage=True).to(device)
    diffusion_model.enable_model_cpu_offload()
    controlnet = diffusion_model.controlnet.to(dtype_half)
    controlnet.requires_grad_(False)
    # components
    tokenizer = diffusion_model.tokenizer
    text_encoder = diffusion_model.text_encoder
    vae = diffusion_model.vae
    unet = diffusion_model.unet.to(dtype_half)
    # save VRAM
    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()
    # set device
    vae = vae.to(device).requires_grad_(False)
    text_encoder = text_encoder.to(device).requires_grad_(False)
    unet = unet.to(device).requires_grad_(False)
    # set scheduler
    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=dtype_half)
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    # set timesteps
    num_train_timesteps = len(scheduler.betas)
    scheduler.set_timesteps(num_train_timesteps)

    # vsd
    unet_phi, unet_lora_layers = extract_lora_diffusers(unet, device)
    
    ######################################################
    # 3. Dataset and learnable model creation                                     
    ######################################################
    # prepare text embeddings
    text_embeddings_1 = encode_prompt_with_a_prompt_and_n_prompt(batch_size, prompt_1, a_prompt, n_prompt, tokenizer, text_encoder, device, particle_num_vsd)
    text_embeddings_2 = encode_prompt_with_a_prompt_and_n_prompt(batch_size, prompt_2, a_prompt, n_prompt, tokenizer, text_encoder, device, particle_num_vsd)
    uncond, cond = text_embeddings_2.chunk(2)

    # get or generate uv parameterization
    # new_mesh = import_smart_uv_mesh(ply_filename, work_dir, device, latent_texture_size, latent_channels, scene_scale)
    # uv_coords_list, uv_coords_eval_list, depth_tensor_list, depth_tensor_eval_list, angle_deviation_list = data_prepare(dtype_half, num_views_eval, num_views, device, dist, at, render_size, faces_per_pixel, new_mesh)

    new_mesh = import_smart_uv_mesh_scannetpp(ply_filename, work_dir, device, latent_texture_size, latent_channels, scene_scale)
    uv_coords_list, depth_tensor_list = data_prepare_scannetpp(dtype_half, cameras, num_views_eval, num_views, device, dist, at, render_size, faces_per_pixel, new_mesh)

    # vsd learnable paramters
    phi_params = list(unet_lora_layers.parameters())
    print(f'number of trainable parameters of phi model in optimizer: {sum(p.numel() for p in phi_params if p.requires_grad)}')
    
    # Hierarchical RGB field
    # texture preparation
    tex_reg_weights = [pow(2, num_hierarchical_layers - i - 1) for i in range(num_hierarchical_layers)]
    tex_reg_weights[-1] = 0

    # particle_models = [
    #     HierarchicalRGB_field(dtype_half, generator, texture_size, num_layers=num_hierarchical_layers).to(device)
    #     for _ in range(particle_num_vsd)
    # ]

    particle_models = [
        Hashgrid_MLP(config['hashgrid_config'], config['renderer_config']).to(device).to(dtype_half)
        for _ in range(particle_num_vsd)
    ]

    # class RGB_Image(nn.Module):
    #     def __init__(self, batch_size, render_size, device):
    #         super().__init__()
    #         self.param = nn.Parameter(
    #             torch.randn(batch_size, 3, render_size, render_size,
    #                         device=device, dtype=torch.float16)
    #         )
    #         self.param.requires_grad = True

    #     def forward(self, *args, **kwargs):
    #         # in case you actually need a forward; otherwise you can omit this
    #         return self.param
    
    # particle_models = [
    #     RGB_Image(batch_size, render_size, device).to(device).to(dtype_half)
    #     for _ in range(particle_num_vsd)
    # ]

    particles = nn.ModuleList(particle_models)

    particles_to_optimize = [param for hashmlp in particles for param in hashmlp.parameters() if param.requires_grad]
    texture_params = [p for p in particles_to_optimize if p.requires_grad]
    print("=> Total number of trainable parameters for texture: {}".format(sum(p.numel() for p in texture_params if p.requires_grad)))

    ######################################################
    # 4. Texture optimization via SDS                                   
    ######################################################
    # t schedule dreamtime
    alphas_cumprod = scheduler.alphas_cumprod.clone().detach()
    alphas_cumprod = alphas_cumprod.cpu().numpy()
    chosen_ts_stage_1 = get_t_schedule_dreamtime(num_train_timesteps, Niter_stage_1 + 1, alphas_cumprod, work_dir)  # important to make num_steps + 1, or the last inference would not be conducted

    chosen_ts_stage_2 = get_t_schedule_dreamtime(num_train_timesteps, Niter_stage_2 + 1, alphas_cumprod, work_dir)

    losses = {
        "rgb": {"weight": 1.0, "values": []},
        "sds": {"weight": 1.0, "values": []}
    }

    # The optimizers
    optimizer = torch.optim.AdamW(particles_to_optimize, lr=lr, eps=eps, weight_decay=weight_decay)
    phi_optimizer = torch.optim.AdamW([{'params': phi_params, 'lr': phi_lr}], lr=phi_lr, eps=1e-4)

    # loss weights
    loss_weights = get_loss_weights(dtype_half, scheduler.betas)

    # DDIM inversion of ref image
    # # should move this before the first stage, if before the second stage, the results would be different from normal results TODO: still debugging
    # img_ref_tensor = load_ref_img(dtype_half, img_ref_path, render_size=render_size, contrast=contrast, add_noise=add_noise, noise_value=noise_value)
    # # reversion trajectory
    # latents_ref_inversion = encode_ddim(diffusion_model, img_ref_tensor, uncond, encode_steps)
    # torch.save(latents_ref_inversion, f'latent_{img_ref_name}.pt')
    # # save latents_ref_inversion
    # ref_sample = 1 / vae.config.scaling_factor * latents_ref_inversion.clone().detach()
    # ref_image_ = vae.decode(ref_sample).sample.to(dtype_half)
    # save_image((ref_image_ / 2 + 0.5).clamp(0, 1), f'{rgb_path_stage_2}/test_{img_ref_save_name}.png')

    ######################################################
    # 5. First stage                                  
    ######################################################
    # loop_stage_1 = tqdm(range(Niter_stage_1))
    # for i in loop_stage_1:
    #     # Initialize optimizer
    #     optimizer.zero_grad()
        
    #     # Losses to smooth /regularize the mesh shape
    #     loss = {k: torch.tensor(0.0, device=device) for k in losses}

    #     chosen_t = chosen_ts_stage_1[i]
    #     t = torch.tensor([chosen_t]).to(device)
        
    #     # Randomly select one view and one target view to optimize over
    #     for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
    #         uv_coords = uv_coords_list[j].to(device)
    #         zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
    #         depth_tensor = depth_tensor_list[j].to(device)

    #         for model in particles:
    #             # get rgb
    #             rgb = get_rgb(uv_coords, zero_mask, model, device)
    #             rgb = rgb.to(dtype_half)

    #             # get target
    #             target = (images[j].to(device).unsqueeze(0).permute(0, 2, 3, 1) - 0.5) * 2
    #             expanded_zero_mask = zero_mask.unsqueeze(-1).expand_as(target)
    #             target = target.masked_fill(expanded_zero_mask, -1)
    #             target = target.permute(0, 3, 1, 2).to(device)
    #             target = target.clamp(-1, 1)
    #             target = target.to(dtype_half)

    #             if latent_rgb_optimize:
    #                 # encode rgb into latent space
    #                 h = vae.encoder(rgb).to(device)
    #                 moments = vae.quant_conv(h).to(device)
    #                 mean, logvar = torch.chunk(moments, 2, dim=1)
    #                 std = torch.exp(0.5 * logvar).to(device)
    #                 sample = mean + std * torch.randn_like(mean).to(device)
    #                 latents_rgb = vae.config.scaling_factor * sample
    #                 # encode target into latent space
    #                 h_ = vae.encoder(target).to(device)
    #                 moments_ = vae.quant_conv(h_).to(device)
    #                 mean_, logvar_ = torch.chunk(moments_, 2, dim=1)
    #                 std_ = torch.exp(0.5 * logvar_).to(device)
    #                 sample_ = mean_ + std_ * torch.randn_like(mean_).to(device)
    #                 latents_target = vae.config.scaling_factor * sample_

    #                 loss_rgb = ((latents_rgb - latents_target) ** 2).mean()
    #             else:
    #                 loss_rgb = ((rgb - target) ** 2).mean()

    #             loss["rgb"] += loss_rgb / num_views_per_iteration

    #         # Weighted sum of the losses
    #         sum_loss = torch.tensor(0.0, device=device)
    #         for k, l in loss.items():
    #             sum_loss += l * losses[k]["weight"]
    #             losses[k]["values"].append(float(l.detach().cpu()))
            
    #         # Optimization step
    #         sum_loss.backward()
    #         optimizer.step()
    #         torch.cuda.empty_cache()

    #         # Print the losses
    #         loop_stage_1.set_description(f"stage: first optimize stage, total_loss = %.6f" % sum_loss)

    #         # save results
    #         if i % plot_period == 0:
    #             rgb_list = []
    #             eval_rgb_list = []
    #             # save image results
    #             for view_eval_id in range(num_views_eval):
    #                 uv_coords = uv_coords_list[view_eval_id].to(device)
    #                 zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
    #                 rgb = get_rgb(uv_coords, zero_mask, model, device)

    #                 rgb_img = rgb.clone().detach().cpu().squeeze(0)
    #                 rgb_list.append(rgb_img)
    #                 eval_img = ((images[view_eval_id] - 0.5) * 2).clone().detach().cpu()
    #                 eval_rgb_list.append(eval_img)

    #             rgb_imgs = torch.stack(rgb_list, dim=0)
    #             eval_imgs = torch.stack(eval_rgb_list, dim=0)

    #             img = torch.cat([rgb_imgs, eval_imgs], dim=2)

    #             save_image((img / 2 + 0.5).clamp(0, 1), rgb_path_stage_1 + f'/rgb_{i}_predict.png')
                
    #             # save texture map using inference
    #             inference(texture_size, particles, device, rgb_path_stage_1, i)

    # # save the last round results
    # rgb_list = []
    # eval_rgb_list = []
    # # save image results
    # for view_eval_id in range(num_views_eval):
    #     uv_coords = uv_coords_list[view_eval_id].to(device)
    #     zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
    #     rgb = get_rgb(uv_coords, zero_mask, model, device)

    #     rgb_img = rgb.clone().detach().cpu().squeeze(0)
    #     rgb_list.append(rgb_img)
    #     eval_img = ((images[view_eval_id] - 0.5) * 2).clone().detach().cpu()
    #     eval_rgb_list.append(eval_img)
    # rgb_imgs = torch.stack(rgb_list, dim=0)
    # eval_imgs = torch.stack(eval_rgb_list, dim=0)

    # img = torch.cat([rgb_imgs, eval_imgs], dim=2)

    # save_image((img / 2 + 0.5).clamp(0, 1), rgb_path_stage_1 + f'/rgb_{i + 1}_predict.png')
    # # save texture map using inference
    # inference(texture_size, particles, device, rgb_path_stage_1, i + 1)

    # # save weights for each checkpoint step
    # for index, texture in enumerate(particles):
    #     checkpoint = {
    #         "texture": texture.state_dict(),
    #     }
    #     torch.save(
    #         checkpoint,
    #         os.path.join(rgb_path_stage_1, f"checkpoint_{i + 1}_particle_{index}.pth")
    #     )

    # # plot losses
    # loss_path = rgb_path_stage_1 + f'/losses.png'
    # plot_losses(losses, loss_path)

    ######################################################
    # 6. Second stage                                  
    ######################################################
    # load model from stage 1 to reduce testing time
    if init_flag:
        output_dir_checkpoint = "./outputs/PTDiffusion_3d_RGB_optimize/2025-05-06_09-56-36_init_True_cs_0.25_br_1.0_tr_0.0_direct_0.4_decayed_0.2_stage1_4000_stage2_500_seed_99_a castle_a photo of a japanese style living room/rgb_uv_stage_1" # checkpoint directory
        checkpoint_name = "checkpoint_4000_particle_0.pth"
        checkpoint_path = os.path.join(output_dir_checkpoint, checkpoint_name)
        checkpoint = torch.load(checkpoint_path)
        for model in particles:
            model.load_state_dict(checkpoint['texture'])
    else:
        pass

    ref_latent = torch.load(f'latent_{img_ref_name}.pt').cuda().to(torch.float16)

    if ref_linear:
        # linear ref timesteps
        ref_update_steps = np.linspace(990, 0, ref_steps)
    else:
        # dreamtime ref timesteps
        ref_update_steps = get_t_schedule_dreamtime(num_train_timesteps, ref_steps + 1, alphas_cumprod, work_dir)

    ref_idx = 0

    weights_decay_transfer = torch.linspace(blending_ratio_default, 0.0, int(decayed_transfer_ratio * Niter_stage_2)) ** exponent

    stage = "direct_transfer"

    if update_linear:
        # linear update timesteps
        chosen_ts_stage_2 = np.linspace(990, 0, Niter_stage_2)
    else:
        # dreamtime update timesteps
        chosen_ts_stage_2 = get_t_schedule_dreamtime(num_train_timesteps, Niter_stage_2 + 1, alphas_cumprod, work_dir)

    loop_stage_2 = tqdm(range(Niter_stage_2))
    for i in loop_stage_2:
        # Initialize optimizer
        optimizer.zero_grad()
        
        # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=device) for k in losses}

        chosen_t = int(chosen_ts_stage_2[i])
        t = torch.tensor([chosen_t]).to(device)

        chosen_t_last = int(chosen_ts_stage_2[i + 1]) if i < Niter_stage_2 - 1 else int(chosen_ts_stage_2[i])
        t_last = torch.tensor([chosen_t_last]).to(device)

        # ref_latent DDIM step
        if i % int(Niter_stage_2 / ref_steps) == 0: # only do DDIM step for ref_steps steps
            # ref chosen t and last t
            chosen_t_ref = int(ref_update_steps[ref_idx])
            t_ref = torch.tensor([chosen_t_ref]).to(device)

            chosen_t_ref_last = int(ref_update_steps[ref_idx + 1]) if ref_idx < ref_steps - 1 else int(ref_update_steps[ref_idx])
            t_ref_last = torch.tensor([chosen_t_ref_last]).to(device)

            ts_tensor_ref = torch.full((ref_latent.size(0),), chosen_t_ref, device=ref_latent.device, dtype=torch.long)

            ref_a_prev, ref_pred_x0, ref_e = p_sample_ddim(
                ref_steps, ref_latent, uncond, ts_tensor_ref, t_ref, t_ref_last,
                scheduler, unet, return_all=True
            )
            # direct
            ref_latent = ref_a_prev.sqrt() * ref_pred_x0 + (1. - ref_a_prev).sqrt() * ref_e
            ref_latent = ref_latent.to(dtype_half)

            ref_idx += 1

        # transfer ratio to make the timesteps lower for phase transfer
        if i < transfer_ratio * Niter_stage_2:
            if i % log_step == 0:
                ref_latent_current = 1 / vae.config.scaling_factor * ref_latent.clone().detach()
                with torch.no_grad():
                    image_ref = vae.decode(ref_latent_current).sample.to(dtype_half)
                save_image((image_ref / 2 + 0.5).clamp(0, 1), f'{rgb_path_stage_2}/ref_{i}_t{t.item()}.png')
            continue
        
        # Randomly select one view and one target view to optimize over
        for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
            j = 5
            uv_coords = uv_coords_list[j].to(device)
            zero_mask = (uv_coords[..., 0] == 0) & (uv_coords[..., 1] == 0)
            depth_tensor = depth_tensor_list[j].to(device)

            for model in particles:
                rgb = get_rgb(uv_coords, zero_mask, model, device)
                # rgb = model().to(device)
                rgb = rgb.to(dtype_half)

                # latents, noise, noisy_latents = get_noisy_latents(vae, scheduler, rgb, t, device)

                # encode rgb into latent space
                h = vae.encoder(rgb).to(device)
                moments = vae.quant_conv(h).to(device)
                mean, logvar = torch.chunk(moments, 2, dim=1)
                std = torch.exp(0.5 * logvar).to(device)
                sample = mean + std * torch.randn_like(mean).to(device)
                latents = vae.config.scaling_factor * sample

                # get target
                target = (images[j].to(device).unsqueeze(0).permute(0, 2, 3, 1) - 0.5) * 2
                expanded_zero_mask = zero_mask.unsqueeze(-1).expand_as(target)
                target = target.masked_fill(expanded_zero_mask, -1)
                target = target.permute(0, 3, 1, 2).to(device)
                target = target.clamp(-1, 1)
                target = target.to(dtype_half)

                # encode target into latent space
                h_ = vae.encoder(target).to(device)
                moments_ = vae.quant_conv(h_).to(device)
                mean_, logvar_ = torch.chunk(moments_, 2, dim=1)
                std_ = torch.exp(0.5 * logvar_).to(device)
                sample_ = mean_ + std_ * torch.randn_like(mean_).to(device)
                latents_target = vae.config.scaling_factor * sample_

                # transfer latents
                if i <= (transfer_ratio + direct_transfer_ratio) * Niter_stage_2:
                    stage = "direct_transfer"
                    blending_ratio = blending_ratio_default
                elif i < (transfer_ratio + direct_transfer_ratio + decayed_transfer_ratio) * Niter_stage_2:
                    stage = "decayed_transfer"
                    blending_ratio = weights_decay_transfer[i - int((transfer_ratio + direct_transfer_ratio) * Niter_stage_2) - 1]
                else:
                    stage = "refining"
                    blending_ratio = 0.0
                    
                latents_transferred = phase_substitute(ref_pred_x0.clone(), latents_target, blending_ratio)

                loss_sds = 0.5 * F.mse_loss(latents, latents_transferred, reduction="mean")

                loss["sds"] += loss_sds / num_views_per_iteration
                loss["rgb"] += 0.0
        
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
            loop_stage_2.set_description(f"stage: {stage}, total_loss = %.6f, br = {blending_ratio}" % sum_loss)

            # save results
            if i % log_step == 0:
                tmp_latents = 1 / vae.config.scaling_factor * latents.clone().detach()
                pred_latents = 1 / vae.config.scaling_factor * latents_transferred.clone().detach()
                ref_latent_current = 1 / vae.config.scaling_factor * ref_latent.clone().detach()
                with torch.no_grad():
                    image_ = vae.decode(tmp_latents).sample.to(dtype_half)
                    image_x0 = vae.decode(pred_latents).sample.to(dtype_half)
                    image_ref = vae.decode(ref_latent_current).sample.to(dtype_half)
                    image = torch.cat((image_, image_x0, depth_tensor, image_ref), dim=2)
                save_image((image / 2 + 0.5).clamp(0, 1), f'{rgb_path_stage_2}/predictions_{i}_t{t.item()}.png')
                save_image((image_ / 2 + 0.5).clamp(0, 1), f'{rgb_path_stage_2}/rgb_{i}_t{t.item()}.png')
                save_image((image_x0 / 2 + 0.5).clamp(0, 1), f'{rgb_path_stage_2}/predicted_x0_{i}_t{t.item()}.png')
                save_image((image_ref / 2 + 0.5).clamp(0, 1), f'{rgb_path_stage_2}/ref_{i}_t{t_ref.item()}.png')

            if i % plot_period == 0:
                # save texture map using inference
                inference(texture_size, particles, device, rgb_path_stage_2, i)

    # save the last round results
    tmp_latents = 1 / vae.config.scaling_factor * latents.clone().detach()
    pred_latents = 1 / vae.config.scaling_factor * latents_transferred.clone().detach()
    ref_latent_current = 1 / vae.config.scaling_factor * ref_latent.clone().detach()
    with torch.no_grad():
        image_ = vae.decode(tmp_latents).sample.to(dtype_half)
        image_x0 = vae.decode(pred_latents).sample.to(dtype_half)
        image_ref = vae.decode(ref_latent_current).sample.to(dtype_half)
        image = torch.cat((image_, image_x0, depth_tensor, image_ref), dim=2)
    save_image((image / 2 + 0.5).clamp(0, 1), f'{rgb_path_stage_2}/predictions_{i}_t{t.item()}.png')
    save_image((image_ / 2 + 0.5).clamp(0, 1), f'{rgb_path_stage_2}/rgb_{i}_t{t.item()}.png')
    save_image((image_x0 / 2 + 0.5).clamp(0, 1), f'{rgb_path_stage_2}/predicted_x0_{i}_t{t.item()}.png')
    save_image((image_ref / 2 + 0.5).clamp(0, 1), f'{rgb_path_stage_2}/ref_{i}_t{t_ref.item()}.png')
    save_image((depth_tensor / 2 + 0.5).clamp(0, 1), f'{rgb_path_stage_2}/depth_tensor.png')
    # save texture map using inference
    inference(texture_size, particles, device, rgb_path_stage_2, i + 1)

    # save weights for each checkpoint step
    for index, texture in enumerate(particles):
        checkpoint = {
            "texture": texture.state_dict(),
        }
        torch.save(
            checkpoint,
            os.path.join(rgb_path_stage_2, f"checkpoint_{i + 1}_particle_{index}.pth")
        )
    unet_phi.save_attn_procs(save_directory=rgb_path_stage_2)

    # plot losses
    loss_path = rgb_path_stage_2 + f'/losses.png'
    plot_losses(losses, loss_path)