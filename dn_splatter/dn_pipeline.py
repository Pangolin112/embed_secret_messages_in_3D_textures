import json
import os
import random
import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal, Optional, Type

import numpy as np
import open3d as o3d
import torch
import trimesh
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch.cuda.amp.grad_scaler import GradScaler

from dn_splatter.data.mushroom_utils.eval_faro import depth_eval_faro
from dn_splatter.dn_model import DNSplatterModelConfig
from dn_splatter.metrics import PDMetrics
from dn_splatter.utils import camera_utils
from dn_splatter.utils.utils import gs_render_dataset_images, ns_render_dataset_images
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
)
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    DDP,
    Model,
    VanillaPipeline,
    VanillaPipelineConfig,
    dist,
)
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE

# imports for secret view editing
import yaml
from types import SimpleNamespace
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import lpips
import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt

from dn_splatter.ip2p_ptd import IP2P_PTD

from dn_splatter.ip2p_depth import InstructPix2Pix_depth

from nerfstudio.engine.callbacks import TrainingCallbackAttributes

from dn_splatter.utils.secret_utils import generate_ves_poses_opengl

from dn_splatter.utils.pie_utils import opencv_seamless_clone

from dn_splatter.utils.edge_loss_utils import SobelFilter, SobelEdgeLoss

import copy

# seva imports
from seva.data_io import get_parser
from seva.eval import (
    IS_TORCH_NIGHTLY,
    create_transforms_simple,
    infer_prior_stats,
    run_one_scene,
)
from seva.geometry import (
    get_default_intrinsics,
    get_preset_pose_fov,
)
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DiscreteDenoiser
from seva.utils import load_model

# lseg
from dn_splatter.utils.lseg_utils import lseg_module_init
from torchmetrics.segmentation import MeanIoU

# sam 2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# clip
import open_clip

# histogram semantic
from dn_splatter.utils.histogram_semantic_utils import (
    compute_histogram_loss_color_segments,
    visualize_segments,
)


def interpolate_to_patch_size(img_bchw, patch_size):
    # Interpolate the image so that H and W are multiples of the patch size
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = torch.nn.functional.interpolate(img_bchw, size=(target_H, target_W))
    return img_bchw, target_H, target_W


@dataclass
class DNSplatterPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: DNSplatterPipeline)
    datamanager: DataManagerConfig = field(default_factory=lambda: DataManagerConfig())
    model: ModelConfig = field(default_factory=lambda: DNSplatterModelConfig())
    experiment_name: str = "experiment"
    """Experiment name for saving metrics and rendered images to disk"""
    skip_point_metrics: bool = True
    """Skip evaluating point cloud metrics"""
    num_pd_points: int = 1_000_000
    """Total number of points to extract from train/eval renders for pointcloud reconstruction"""
    save_train_images: bool = False
    """saving train images to disc"""
    gs_steps: int = 2500
    """how many GS steps between dataset updates"""


class DNSplatterPipeline(VanillaPipeline):
    """Pipeline for convenient eval metrics across model types"""

    def __init__(
        self,
        config: DNSplatterPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.datamanager.to(device)

        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz"
            in self.datamanager.train_dataparser_outputs.metadata  # type: ignore
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata[
                "points3D_xyz"
            ]  # type: ignore
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata[
                "points3D_rgb"
            ]  # type: ignore
            if "points3D_normals" in self.datamanager.train_dataparser_outputs.metadata:
                normals = self.datamanager.train_dataparser_outputs.metadata[
                    "points3D_normals"
                ]  # type: ignore
                seed_pts = (pts, pts_rgb, normals)
            else:
                seed_pts = (pts, pts_rgb)

        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                Model,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])

        self.pd_metrics = PDMetrics()

        ######################################################
        # Secret view updating                              
        ######################################################
        # prepare for secret view editing
        # which image index we are editing
        self.curr_edit_idx = 0
        # whether we are doing regular GS updates or editing images
        self.makeSequentialEdits = False
        
        # whether we are doing the first sequential edition
        self.first_SequentialEdit = True

        # whether we are at the first step
        self.first_step = True

        self.secret_loss_weight: float = 5.0

        config_path = 'config/config.yaml'
        with open(config_path, 'r') as file:
            cfg_dict = yaml.safe_load(file)
        self.config_secret = SimpleNamespace(**cfg_dict)

        self.config_secret.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

        self.ip2p_ptd = IP2P_PTD(
            self.dtype, 
            self.config_secret.device, 
            conditioning_scale = self.config_secret.conditioning_scale,
            prompt=self.config_secret.prompt_2, 
            a_prompt=self.config_secret.a_prompt,
            n_prompt=self.config_secret.n_prompt,
            t_dec=self.config_secret.t_dec, 
            image_guidance_scale=self.config_secret.image_guidance_scale_ip2p_ptd, 
            async_ahead_steps=self.config_secret.async_ahead_steps
        )

        self.text_embeddings_ip2p = self.ip2p_ptd.text_embeddings_ip2p

        self.ip2p_depth = InstructPix2Pix_depth(
            self.dtype, 
            self.config_secret.device, 
            self.config_secret.render_size, 
            self.config_secret.conditioning_scale
        )

        # refenece image for lpips computing
        self.ref_image = Image.open(self.ip2p_ptd.ref_img_path).convert('RGB').resize((self.config_secret.render_size, self.config_secret.render_size))
        # Convert reference image to tensor and process it the same way
        ref_image_tensor = torch.from_numpy(np.array(self.ref_image)).float() / 255.0  # Convert to [0, 1] range
        ref_image_tensor = ref_image_tensor.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
        self.ref_image_tensor = (ref_image_tensor * 2 - 1).clamp(-1, 1).to(self.config_secret.device)  # Convert to [-1, 1] range

        # 'vgg', 'alex', 'squeeze', lpips loss
        self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(self.config_secret.device)

        # edge loss 
        self.edge_loss_fn = SobelEdgeLoss(loss_type='l1', ksize=3, use_grayscale=self.config_secret.use_grayscale)

        # secret data preparation
        secret_view_idx = self.config_secret.secret_view_idx
        self.camera_secret, self.data_secret = self.datamanager.next_train_idx(secret_view_idx)
        # 1st stage secret view rendering
        self.original_image_secret = self.datamanager.original_cached_train[secret_view_idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        self.depth_image_secret = self.datamanager.original_cached_train[secret_view_idx]["depth"] # [bs, h, w]
        # original secret edges
        self.original_secret_edges = SobelFilter(ksize=3, use_grayscale=self.config_secret.use_grayscale)(self.original_image_secret)
        
        # # lseg model
        self.lseg_model = lseg_module_init()
        with torch.no_grad():
            self.original_image_sem_feature = self.lseg_model.get_image_features(self.original_image_secret.to(self.config_secret.device))

        # # sam2 model
        # self.sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
        # # ref: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_image_predictor.py#L117
        # with torch.no_grad():
        #     # input image should be of size 1x3xHxW
        #     backbone_out = self.sam2_predictor.model.forward_image(self.original_image_secret.to(self.config_secret.device))
        #     _, vision_feats, _, _ = self.sam2_predictor.model._prepare_backbone_features(backbone_out)
        #     # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        #     if self.sam2_predictor.model.directly_add_no_mem_embed:
        #         vision_feats[-1] = vision_feats[-1] + self.sam2_predictor.model.no_mem_embed

        #     feats = [
        #         feat.permute(1, 2, 0).reshape(1, -1, *feat_size)
        #         for feat, feat_size in zip(vision_feats[::-1], self.sam2_predictor._bb_feat_sizes[::-1])
        #     ][::-1]
        #     self.original_image_sam2_feature = feats[-1]

        # # dinov2 model
        # print("Loading DINOv2 model...")
        # dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        # self.dinov2 = dinov2.to(self.config_secret.device)

        # image, target_H, target_W = interpolate_to_patch_size(self.original_image_secret, self.dinov2.patch_size)
        # image = image.cuda()
        # with torch.no_grad():
        #     features = self.dinov2.forward_features(image)["x_norm_patchtokens"][0]

        #     features = features.cpu()

        #     features_hwc = features.reshape((target_H // self.dinov2.patch_size, target_W // self.dinov2.patch_size, -1))
        #     features_chw = features_hwc.permute((2, 0, 1))
        #     self.original_image_dinov2_feature = features_chw.to(self.config_secret.device)

        # semantic map
        # self.semantic_map_secret = self.data_secret["semantic"].permute(0, 3, 1, 2) # [1, c, h, w]

        # print("\n semantic map values: \n", self.semantic_map_secret[0])

        # miou metric
        self.miou = MeanIoU(
            num_classes=9,
            include_background=False,
            per_class=True,
            input_format= "index")
        self.miou = self.miou.to(self.config_secret.device)

        # second secret view preparation
        secret_view_idx_2 = self.config_secret.secret_view_idx_2
        self.camera_secret_2, self.data_secret_2 = self.datamanager.next_train_idx(secret_view_idx_2)
        self.original_image_secret_2 = self.datamanager.original_cached_train[secret_view_idx_2]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        self.depth_image_secret_2 = self.datamanager.original_cached_train[secret_view_idx_2]["depth"] # [bs, h, w]

        self.first_iter = True

        c2w_secret = np.concatenate(
            [
                self.camera_secret.camera_to_worlds.cpu().numpy()[0],
                np.array([[0, 0, 0, 1]], dtype=np.float32)  # Add the last row for homogeneous coordinates
            ],
            0,
        )

        # generate ves cameras for validating the seva's ability
        self.ves_c2ws = generate_ves_poses_opengl(
            c2w_secret, 
            angle_limit_degrees=self.config_secret.angle_limits[0]
        )

        # Define VES view indices - these will be appended after existing training views
        self.num_ves_views = len(self.ves_c2ws)  # Should be 9 based on your code
        original_train_size = len(self.datamanager.cached_train)
        self.ves_view_indices = list(range(original_train_size, original_train_size + self.num_ves_views))
        
        # ############### Extend cached_train to accommodate VES views ###############
        # Create placeholder entries for VES views
        # for i in range(self.num_ves_views):
        #     # Create a placeholder data entry with the same structure as existing entries
        #     # You may need to adjust this based on your actual data structure
        #     placeholder_entry = {
        #         "image": torch.zeros_like(self.datamanager.cached_train[0]["image"]),  # Placeholder image
        #         "idx": original_train_size + i,
        #         "is_ves_view": True,  # Flag to identify VES views
        #     }
            
        #     # Add other required fields from your data structure but set depth/normal to None for VES views
        #     if "depth" in self.datamanager.cached_train[0]:
        #         placeholder_entry["depth"] = None  # Set to None instead of zeros
            
        #     if "sensor_depth" in self.datamanager.cached_train[0]:
        #         placeholder_entry["sensor_depth"] = None
                
        #     if "mono_depth" in self.datamanager.cached_train[0]:
        #         placeholder_entry["mono_depth"] = None
                
        #     if "normal" in self.datamanager.cached_train[0]:
        #         placeholder_entry["normal"] = None
                
        #     if "confidence" in self.datamanager.cached_train[0]:
        #         placeholder_entry["confidence"] = None
            
        #     # Add any other fields that exist in your cached_train entries
        #     for key in self.datamanager.cached_train[0].keys():
        #         if key not in placeholder_entry:
        #             if key in ["depth", "sensor_depth", "mono_depth", "normal", "confidence"]:
        #                 placeholder_entry[key] = None
        #             elif isinstance(self.datamanager.cached_train[0][key], torch.Tensor):
        #                 placeholder_entry[key] = torch.zeros_like(self.datamanager.cached_train[0][key])
        #             else:
        #                 placeholder_entry[key] = self.datamanager.cached_train[0][key]  # Copy non-tensor values
            
        #     self.datamanager.cached_train.append(placeholder_entry)
        
        # # Also extend original_cached_train if it exists
        # if hasattr(self.datamanager, 'original_cached_train'):
        #     for i in range(self.num_ves_views):
        #         placeholder_entry = {
        #             "image": torch.zeros_like(self.datamanager.original_cached_train[0]["image"]),
        #             "idx": original_train_size + i,
        #             "is_ves_view": True,
        #         }
                
        #         if "depth" in self.datamanager.original_cached_train[0]:
        #             placeholder_entry["depth"] = None
                
        #         for key in self.datamanager.original_cached_train[0].keys():
        #             if key not in placeholder_entry:
        #                 if key in ["depth", "sensor_depth", "mono_depth", "normal", "confidence"]:
        #                     placeholder_entry[key] = None
        #                 elif isinstance(self.datamanager.original_cached_train[0][key], torch.Tensor):
        #                     placeholder_entry[key] = torch.zeros_like(self.datamanager.original_cached_train[0][key])
        #                 else:
        #                     placeholder_entry[key] = self.datamanager.original_cached_train[0][key]
                
        #         self.datamanager.original_cached_train.append(placeholder_entry)

        # ############### generate ves cameras ###############
        # self.ves_cameras = []
        # for ves_c2w in self.ves_c2ws:
        #     # need to create a new copy for each camera, or all the cameras will refer to the same object
        #     camera_secret_copy = copy.deepcopy(self.camera_secret)
        #     ves_c2w_tensor = torch.tensor(ves_c2w, dtype=torch.float32, device=self.config_secret.device)
        #     camera_secret_copy.camera_to_worlds = ves_c2w_tensor[:3, :4].unsqueeze(0)
        #     self.ves_cameras.append(camera_secret_copy)

        # # seva c2ws input
        # self.task = "img2trajvid_s-prob"

        # # convert from OpenGL to OpenCV camera format
        # self.seva_c2ws = np.stack(self.ves_c2ws, axis=0) @ np.diag([1, -1, -1, 1])

        # DEFAULT_FOV_RAD = 0.9424777960769379  # 54 degrees by default
        # self.num_frames = 9
        # fovs = np.full((self.num_frames,), DEFAULT_FOV_RAD)
        # aspect_ratio = 1.0
        # Ks = get_default_intrinsics(fovs, aspect_ratio=aspect_ratio)  # unormalized
        # Ks[:, :2] *= (
        #     torch.tensor([self.config_secret.render_size, self.config_secret.render_size]).reshape(1, -1, 1).repeat(Ks.shape[0], 1, 1)
        # )  # normalized
        # self.Ks = Ks.numpy()

        # # model loading
        # if IS_TORCH_NIGHTLY:
        #     COMPILE = True
        #     os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
        #     os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        # else:
        #     COMPILE = False
        # version=1.1
        # pretrained_model_name_or_path="stabilityai/stable-virtual-camera"
        # weight_name="model.safetensors"
        # self.MODEL = SGMWrapper(
        #     load_model(
        #         model_version=version,
        #         pretrained_model_name_or_path=pretrained_model_name_or_path,
        #         weight_name=weight_name,
        #         device="cpu",
        #         verbose=True,
        #     ).eval()
        # ).to(self.config_secret.device)

        # if COMPILE:
        #     MODEL = torch.compile(MODEL, dynamic=False)

        # self.AE = AutoEncoder(chunk_size=1).to(self.config_secret.device)
        # self.CONDITIONER = CLIPConditioner().to(self.config_secret.device)
        # self.DENOISER = DiscreteDenoiser(num_idx=1000, device=self.config_secret.device)

        # if COMPILE:
        #     self.CONDITIONER = torch.compile(self.CONDITIONER, dynamic=False)
        #     self.AE = torch.compile(self.AE, dynamic=False)

        # self.seed = 23

        # options = {
        #     'chunk_strategy': 'interp', 
        #     'video_save_fps': 30.0, 
        #     'beta_linear_start': 5e-06, 
        #     'log_snr_shift': 2.4, 
        #     'guider_types': 1, 
        #     'cfg': (4.0, 2.0), 
        #     'camera_scale': 0.1, 
        #     'num_steps': 20, 
        #     'cfg_min': 1.2, 
        #     'encoding_t': 1, 
        #     'decoding_t': 1, 
        #     'replace_or_include_input': True, 
        #     'traj_prior': 'stabilization', 
        #     'guider': (1, 2), 
        #     'num_targets': 8
        # }

        # self.VERSION_DICT = {
        #     'H': 512, 
        #     'W': 512, 
        #     'T': 21, 
        #     'C': 4, 
        #     'f': 8,
        #     "options": options,
        # }

        # self.num_inputs = 1
        # self.num_targets = self.num_frames - 1
        # self.input_indices = [0]
        # num_anchors = infer_prior_stats(
        #     self.VERSION_DICT["T"],
        #     self.num_inputs,
        #     num_total_frames=self.num_targets,
        #     version_dict=self.VERSION_DICT,
        # )
        # self.anchor_indices = np.linspace(1, self.num_targets, num_anchors).tolist()

        # self.anchor_c2ws = self.seva_c2ws[[round(ind) for ind in self.anchor_indices]]
        # self.anchor_Ks = self.Ks[[round(ind) for ind in self.anchor_indices]]

        # self.anchor_c2ws = torch.tensor(self.anchor_c2ws[:, :3]).float()
        # self.anchor_Ks = torch.tensor(self.anchor_Ks).float()

        # self.seva_c2ws = torch.tensor(self.seva_c2ws[:, :3]).float()
        # self.Ks = torch.tensor(self.Ks).float()

        
        ######################################################
    
    # add callback function to fetch the components from other parts of the training process.
    def get_training_callbacks(self, attrs: TrainingCallbackAttributes):
        # stash a reference to the Trainer
        self.trainer = attrs.trainer
        # now return whatever callbacks the base class wants
        return super().get_training_callbacks(attrs)

    def _axis_angle_to_rotation_matrix(self, axis_angle):
        """Convert axis-angle to rotation matrix using matrix exponential."""
        # Create skew-symmetric matrix
        K = torch.zeros(3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
        K[0, 1] = -axis_angle[2]
        K[0, 2] = axis_angle[1]
        K[1, 0] = axis_angle[2]
        K[1, 2] = -axis_angle[0]
        K[2, 0] = -axis_angle[1]
        K[2, 1] = axis_angle[0]
        
        # Use matrix exponential (more stable for gradients)
        R = torch.matrix_exp(K)
        return R

    ### 2-stage method (baseline)
    ######################################################
    # start: 2nd stage: IGS2GS + IN2N + pie            
    ######################################################
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_pie_secret_image_cond_{self.config_secret.image_guidance_scale_ip2p_ptd}_async_{self.config_secret.async_ahead_steps}_contrast_{self.ip2p_ptd.contrast}_non_secret_{self.config_secret.image_guidance_scale_ip2p}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True
    #         # dataset downsampling and return the new secret idx
    #         self.config_secret.secret_view_idx = self.datamanager.downsample_dataset(self.config_secret.downsample_factor, self.config_secret.secret_view_idx)

    #     if (not self.makeSequentialEdits):
    #         # the implementation of randomly selecting an index to edit instead of update all images at once
    #         # generate the indexes for non-secret view editing
    #         all_indices = np.arange(len(self.datamanager.cached_train))
    #         allowed = all_indices[all_indices != self.config_secret.secret_view_idx]

    #         if step % self.config_secret.edit_rate == 0:
    #             #----------------non-secret view editing----------------
    #             # randomly select an index to edit
    #             idx = random.choice(allowed)
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #             original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #             depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 50 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # ###########################################
    #                 # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone

    #                 edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                     self.text_embeddings_ip2p.to(self.config_secret.device),
    #                     rendered_image_secret.to(self.dtype),
    #                     self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     False, # is depth tensor
    #                     self.depth_image_secret,
    #                     guidance_scale=self.config_secret.guidance_scale,
    #                     image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                     diffusion_steps=self.config_secret.t_dec,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound,
    #                 )

    #                 # edited_image_secret is B, C, H, W in [0, 1]
    #                 edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #                 # Convert edited_image_target to numpy (NEW TARGET)
    #                 edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #                 # Convert mask to numpy
    #                 if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                     # It's a PyTorch tensor
    #                     mask_tensor = self.ip2p_ptd.mask
    #                     if mask_tensor.dim() == 4:
    #                         mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                     elif mask_tensor.dim() == 3:
    #                         mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                     else:
    #                         mask_np = mask_tensor.cpu().numpy()
    #                     mask_np = (mask_np * 255).astype(np.uint8)
    #                 elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                     # It's a PIL Image
    #                     mask_np = np.array(self.ip2p_ptd.mask)
    #                     # Convert to grayscale if needed
    #                     if mask_np.ndim == 3:
    #                         mask_np = mask_np[:, :, 0]  # Take first channel
    #                     # Ensure it's uint8
    #                     if mask_np.dtype != np.uint8:
    #                         if mask_np.max() <= 1.0:
    #                             mask_np = (mask_np * 255).astype(np.uint8)
    #                         else:
    #                             mask_np = mask_np.astype(np.uint8)
    #                 else:
    #                     # If it's already numpy, just use it
    #                     mask_np = self.ip2p_ptd.mask

    #                 # Call the original opencv_seamless_clone function with numpy arrays
    #                 result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #                 # Convert the result back to PyTorch tensor format
    #                 # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #                 edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #                 edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #                 edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #                 # ###########################################

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v
                    
    #                 # compute lpips score between ref image and current secret rendering
    #                 lpips_score = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 )
    #                 # print(f"lpips score: {lpips_score.item():.6f}")

    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
   
    #         else:
    #             # non-editing steps loss computing
    #             camera, data = self.datamanager.next_train(step)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             # also update the secret view w/o editing (important)
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.config_secret.secret_view_idx:
    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 25 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # edit image using IP2P + PTD when idx == secret_view_idx
    #         if idx == self.config_secret.secret_view_idx:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 secret_idx=0,
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #             # ###########################################
    #             # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #             edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image_secret.to(self.dtype),
    #                 self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 self.depth_image_secret,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             save_image((edited_image_target.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_target_image.png')

    #             # edited_image_secret is B, C, H, W in [0, 1]
    #             edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #             # Convert edited_image_target to numpy (NEW TARGET)
    #             edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #             # Convert mask to numpy
    #             if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                 # It's a PyTorch tensor
    #                 mask_tensor = self.ip2p_ptd.mask
    #                 if mask_tensor.dim() == 4:
    #                     mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                 elif mask_tensor.dim() == 3:
    #                     mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                 else:
    #                     mask_np = mask_tensor.cpu().numpy()
    #                 mask_np = (mask_np * 255).astype(np.uint8)
    #             elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                 # It's a PIL Image
    #                 mask_np = np.array(self.ip2p_ptd.mask)
    #                 # Convert to grayscale if needed
    #                 if mask_np.ndim == 3:
    #                     mask_np = mask_np[:, :, 0]  # Take first channel
    #                 # Ensure it's uint8
    #                 if mask_np.dtype != np.uint8:
    #                     if mask_np.max() <= 1.0:
    #                         mask_np = (mask_np * 255).astype(np.uint8)
    #                     else:
    #                         mask_np = mask_np.astype(np.uint8)
    #             else:
    #                 # If it's already numpy, just use it
    #                 mask_np = self.ip2p_ptd.mask

    #             # Call the original opencv_seamless_clone function with numpy arrays
    #             result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #             # Convert the result back to PyTorch tensor format
    #             # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #             edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #             edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #             edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #             # ###########################################

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # compute lpips score between ref image and current secret rendering
    #             lpips_score = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             )
    #             print(f"lpips score: {lpips_score.item():.6f}")

    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
    ######################################################
    # end: 2nd stage: IGS2GS + IN2N + pie            
    ######################################################


    ### 3-stage method
    ############################################################################################################
    # start: 2nd stage: secret + non-secret loss + masked fighting (ref, original) loss
    ############################################################################################################
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"3_stage_2nd_images_only_secret_loss_fighting_ref_original_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # these constraints are too strong, we need a soft regularization
    #     # term for opacity and mean updating, to make the editing strength
    #     # strong enough.
    #     # self.model.means.requires_grad_(False)
    #     # # self.model.scales.requires_grad_(False)
    #     # self.model.opacities.requires_grad_(False)
    #     # # self.model.quats.requires_grad_(False)

    #     if self.first_step:
    #         self.first_step = False
    #         self.original_means = self.model.means.clone().detach()
    #         self.original_opacities = self.model.opacities.clone().detach()

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)
            
    #     # update the secret view every secret_edit_rate steps
    #     if step % self.config_secret.secret_edit_rate == 0:
    #     # if step % self.config_secret.secret_update_rate == 0:
    #         model_outputs_secret = self.model(self.camera_secret)
    #         metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #         loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret) # l1, lpips, regularization

    #         # compute masked lpips value
    #         mask_np = self.ip2p_ptd.mask
    #         # Convert mask to tensor and ensure it's the right shape/device
    #         mask_tensor = torch.from_numpy(mask_np).float()
    #         if len(mask_tensor.shape) == 2:
    #             mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #         if mask_tensor.shape[0] == 1:
    #             mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #         mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #         mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #         # Prepare model output
    #         model_rgb_secret = (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

    #         # Apply mask to both images
    #         masked_model_rgb = model_rgb_secret * mask_tensor
    #         masked_ref_image = self.ref_image_tensor * mask_tensor

    #         # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #         ref_loss = self.lpips_loss_fn(
    #             masked_model_rgb,
    #             masked_ref_image
    #         ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #         ref_l1_loss = torch.nn.functional.l1_loss(
    #             masked_model_rgb,
    #             masked_ref_image
    #         )
    #         loss_dict_secret["main_loss"] += self.config_secret.ref_loss_weight * ref_loss + ref_l1_loss

    #         # edge loss
    #         # rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #         # edge_loss = self.edge_loss_fn(
    #         #     rendered_image_secret.to(self.config_secret.device), 
    #         #     self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #         #     self.original_secret_edges.to(self.config_secret.device),
    #         #     image_dir,
    #         #     step
    #         # )
    #         # loss_dict_secret["main_loss"] += edge_loss

    #         if step % 100 == 0:
    #             image_save_secret = torch.cat([model_outputs_secret["rgb"].detach().permute(2, 0, 1).unsqueeze(0), ((masked_ref_image + 1) / 2).to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #         # put the secret metrics and loss into the main dict
    #         for k, v in metrics_dict_secret.items():
    #             metrics_dict[f"secret_{k}"] = v
    #         for k, v in loss_dict_secret.items():
    #             loss_dict[f"secret_{k}"] = v

    #     return model_outputs, loss_dict, metrics_dict
    ############################################################################################################
    # end: 2nd stage: secret + non-secret loss + masked fighting (ref, original) loss
    ############################################################################################################

    ############################################################################################################################################
    # start: 3rd stage: dataset downsampling + trust first secret edition only + non-secret editing with ip2p (+ masked edge loss + lseg loss)
    ############################################################################################################################################
    def get_train_loss_dict(self, step: int):
        base_dir = self.trainer.base_dir
        image_dir = base_dir / f"3_stage_3rd_images_only_editing_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
        if not image_dir.exists():
            image_dir.mkdir(parents=True, exist_ok=True)

        # need to set this for each training step
        # disable means, scales, opacities and quats' updating to prevent floaters
        if self.config_secret.disable_floater:
            self.model.means.requires_grad_(False)
            # self.model.scales.requires_grad_(False)
            self.model.opacities.requires_grad_(False)
            # self.model.quats.requires_grad_(False)

        # replace the original dataset with current rendering
        if self.first_step:
            self.first_step = False
        
            for idx in tqdm(range(len(self.datamanager.cached_train))):
                camera, data = self.datamanager.next_train_idx(idx)
                model_outputs = self.model(camera)

                rendered_image = model_outputs["rgb"].detach()

                self.datamanager.original_cached_train[idx]["image"] = rendered_image
                self.datamanager.cached_train[idx]["image"] = rendered_image
                data["image"] = rendered_image

            print("dataset replacement complete!")

            # dataset downsampling and return the new secret idx
            self.config_secret.secret_view_idx = self.datamanager.downsample_dataset(self.config_secret.downsample_factor, self.config_secret.secret_view_idx)
            # # save original semantic map
            # original_image_logits = self.lseg_model.decode_feature(self.original_image_sem_feature)
            # original_semantic = self.lseg_model.visualize_sem(original_image_logits) # (c, h, w)
            # save_image(original_semantic, image_dir / f'{step}_original_semantic.png')
            # save_image(self.original_image_secret, image_dir / f'{step}_original_image.png')

        # start editing
        if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
            self.makeSequentialEdits = True

        # ordinary editing
        if (not self.makeSequentialEdits):
            all_indices = np.arange(len(self.datamanager.cached_train))
            idx = random.choice(all_indices)

            # only edit the non-secret views, since we fully trust the first edition of the secret view
            if step % self.config_secret.edit_rate == 0 and idx != self.config_secret.secret_view_idx:
                #----------------non-secret view editing----------------
                # randomly select an index to edit
                camera, data = self.datamanager.next_train_idx(idx)
                model_outputs = self.model(camera)
                metrics_dict = self.model.get_metrics_dict(model_outputs, data)

                # without editing
                original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
                rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

                depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

                edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
                    self.text_embeddings_ip2p.to(self.config_secret.device),
                    rendered_image.to(self.dtype),
                    original_image.to(self.config_secret.device).to(self.dtype),
                    False, # is depth tensor
                    depth_image,
                    guidance_scale=self.config_secret.guidance_scale,
                    image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
                    diffusion_steps=self.config_secret.t_dec,
                    lower_bound=self.config_secret.lower_bound,
                    upper_bound=self.config_secret.upper_bound,
                )

                # resize to original image size (often not necessary)
                if (edited_image.size() != rendered_image.size()):
                    edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

                # write edited image to dataloader
                edited_image = edited_image.to(original_image.dtype)
                self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
                data["image"] = edited_image.squeeze().permute(1,2,0)

                # save edited non-secret image
                if step % 50 == 0:
                    image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
                    save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

                loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

                ############################ update secret view fully ########################
                # model_outputs_secret = self.model(self.camera_secret)
                # metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
                # loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret) # l1, lpips, regularization
                # # put the secret metrics and loss into the main dict
                # for k, v in metrics_dict_secret.items():
                #     metrics_dict[f"secret_{k}"] = v
                # for k, v in loss_dict_secret.items():
                #     loss_dict[f"secret_{k}"] = v
                # if step % 50 == 0:
                #     save_image((model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2).detach()).clamp(0, 1), image_dir / f'{step}_secret_list.png')
                ############################ update secret view fully ########################

                ############################ prepare secret rendering ######################## 
                model_outputs_secret = self.model(self.camera_secret)
                rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
                ############################ prepare secret rendering ######################## 

                ############################ for edge loss ########################
                # compute masked lpips value
                mask_np = self.ip2p_ptd.mask
                # Convert mask to tensor and ensure it's the right shape/device
                mask_tensor = torch.from_numpy(mask_np).float()
                if len(mask_tensor.shape) == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
                if mask_tensor.shape[0] == 1:
                    mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
                mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
                mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

                # Apply mask to both images
                masked_model_rgb = rendered_image_secret * mask_tensor

                edge_loss = self.edge_loss_fn(
                    masked_model_rgb.to(self.config_secret.device), 
                    self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
                    self.original_secret_edges.to(self.config_secret.device),
                    image_dir,
                    step,
                    mask_tensor
                )

                metrics_dict["secret_edge_loss"] = edge_loss * self.config_secret.edge_loss_weight
                loss_dict["secret_edge_loss"] = edge_loss * self.config_secret.edge_loss_weight
                ############################ for edge loss ########################

                # ############################ for lseg loss ########################
                # # [1, 512, 256, 256]
                # rendered_image_sem_feature = self.lseg_model.get_image_features(rendered_image_secret.to(self.config_secret.device))
                # rendered_image_logits = self.lseg_model.decode_feature(rendered_image_sem_feature)
                # rendered_semantic = self.lseg_model.visualize_sem(rendered_image_logits) # (c, h, w)
                # if step % 50 == 0:
                #     save_image(rendered_semantic, image_dir / f'{step}_rendered_semantic.png')

                # # l1 loss
                # # lseg_loss = torch.nn.functional.l1_loss(rendered_image_sem_feature, self.original_image_sem_feature)
                # # cross loss
                # lseg_loss = (1 - torch.nn.functional.cosine_similarity(rendered_image_sem_feature, self.original_image_sem_feature, dim=1)).mean()

                # metrics_dict["secret_lseg_loss"] = lseg_loss * self.config_secret.lseg_loss_weight
                # loss_dict["secret_lseg_loss"] = lseg_loss * self.config_secret.lseg_loss_weight
                # ############################ for lseg loss ########################

                # ############################ for sam2 loss ########################
                # backbone_out = self.sam2_predictor.model.forward_image(rendered_image_secret.to(self.config_secret.device))
                # _, vision_feats, _, _ = self.sam2_predictor.model._prepare_backbone_features(backbone_out)
                # # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
                # if self.sam2_predictor.model.directly_add_no_mem_embed:
                #     vision_feats[-1] = vision_feats[-1] + self.sam2_predictor.model.no_mem_embed

                # feats = [
                #     feat.permute(1, 2, 0).reshape(1, -1, *feat_size)
                #     for feat, feat_size in zip(vision_feats[::-1], self.sam2_predictor._bb_feat_sizes[::-1])
                # ][::-1]
                # rendered_image_sam2_feature = feats[-1]

                # # l1 loss
                # # lseg_loss = torch.nn.functional.l1_loss(rendered_image_sem_feature, self.original_image_sem_feature)
                # # cross loss
                # sam2_loss = (1 - torch.nn.functional.cosine_similarity(rendered_image_sam2_feature, self.original_image_sam2_feature, dim=1)).mean()

                # metrics_dict["secret_sam2_loss"] = sam2_loss * self.config_secret.sam2_loss_weight
                # loss_dict["secret_sam2_loss"] = sam2_loss * self.config_secret.sam2_loss_weight
                # ############################ for sam2 loss ########################

                # ############################ for dinov2 loss ########################
                # image_dinov2, target_H, target_W = interpolate_to_patch_size(rendered_image_secret, self.dinov2.patch_size)
                # image_dinov2 = image_dinov2.cuda()
                # features = self.dinov2.forward_features(image_dinov2)["x_norm_patchtokens"][0]

                # features_hwc = features.reshape((target_H // self.dinov2.patch_size, target_W // self.dinov2.patch_size, -1))
                # features_chw = features_hwc.permute((2, 0, 1))
                # rendered_image_dinov2_feature = features_chw.to(self.config_secret.device)

                # # l1 loss
                # # dinov2_loss = torch.nn.functional.l1_loss(rendered_image_dinov2_feature, self.original_image_dinov2_feature)
                # # cross loss
                # dinov2_loss = (1 - torch.nn.functional.cosine_similarity(rendered_image_dinov2_feature, self.original_image_dinov2_feature, dim=1)).mean()

                # metrics_dict["secret_dinov2_loss"] = dinov2_loss * self.config_secret.dinov2_loss_weight
                # loss_dict["secret_dinov2_loss"] = dinov2_loss * self.config_secret.dinov2_loss_weight
                # ############################ for dinov2 loss ########################

                # ############################ for semantic histogram loss ########################
                # # visualization = visualize_segments(self.semantic_map_secret)
                # semantic_loss, segment_info = compute_histogram_loss_color_segments(
                #     self.semantic_map_secret, rendered_image_secret, self.original_image_secret.to(self.config_secret.device),
                #     distance_type='wasserstein'
                # )
                
                # metrics_dict["secret_semantic_histogram_loss"] = semantic_loss * self.config_secret.semantic_histogram_loss_weight
                # loss_dict["secret_semantic_histogram_loss"] = semantic_loss * self.config_secret.semantic_histogram_loss_weight
                # ############################ for semantic histogram loss ########################

            # regular updating after editing
            else:
                # non-editing steps loss computing
                camera, data = self.datamanager.next_train(step)
                model_outputs = self.model(camera)
                metrics_dict = self.model.get_metrics_dict(model_outputs, data)
                loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

                # update secret view fully
                model_outputs_secret = self.model(self.camera_secret)
                metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
                loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret) # l1, lpips, regularization
                # put the secret metrics and loss into the main dict
                for k, v in metrics_dict_secret.items():
                    metrics_dict[f"secret_{k}"] = v
                for k, v in loss_dict_secret.items():
                    loss_dict[f"secret_{k}"] = v
                if (step + 1) % 100 == 0:
                    save_image((model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2).detach()).clamp(0, 1), image_dir / f'{step}_secret_rendering.png')

        else:
            # get index
            idx = self.curr_edit_idx
            camera, data = self.datamanager.next_train_idx(idx)
            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)

            original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
            rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

            depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
            edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
                self.text_embeddings_ip2p.to(self.config_secret.device),
                rendered_image.to(self.dtype),
                original_image.to(self.config_secret.device).to(self.dtype),
                False, # is depth tensor
                depth_image,
                guidance_scale=self.config_secret.guidance_scale,
                image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
                diffusion_steps=self.config_secret.t_dec,
                lower_bound=self.config_secret.lower_bound,
                upper_bound=self.config_secret.upper_bound,
            )

            # resize to original image size (often not necessary)
            if (edited_image.size() != rendered_image.size()):
                edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

            # write edited image to dataloader
            edited_image = edited_image.to(original_image.dtype)
            self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
            data["image"] = edited_image.squeeze().permute(1,2,0)

            # save edited non-secret image
            if step % 25 == 0:
                image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
                save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')
            
            if idx == self.config_secret.secret_view_idx:
                self.camera_secret, self.data_secret = self.datamanager.next_train_idx(self.config_secret.secret_view_idx)
                image_save_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
                save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

            loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

            #increment curr edit idx
            # and update all the images in the dataset
            self.curr_edit_idx += 1
            # self.makeSequentialEdits = False
            if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
                self.curr_edit_idx = 0
                self.makeSequentialEdits = False
                self.first_SequentialEdit = False

        return model_outputs, loss_dict, metrics_dict
    ############################################################################################################################################
    # end: 3rd stage: dataset downsampling + trust first secret editiononly + non-secret editing with ip2p (+ masked edge loss + lseg loss)
    ############################################################################################################################################


    @profiler.time_function
    def get_average_eval_image_metrics(
        self,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        assert isinstance(
            self.datamanager,
            (VanillaDataManager, ParallelDataManager, FullImageDatamanager),
        )
        num_eval = len(self.datamanager.fixed_indices_eval_dataloader)
        num_train = len(self.datamanager.train_dataset)  # type: ignore
        all_images = num_train + num_eval

        if not self.config.skip_point_metrics:
            pixels_per_frame = int(
                self.datamanager.train_dataset.cameras[0].width
                * self.datamanager.train_dataset.cameras[0].height
            )
            samples_per_frame = (self.config.num_pd_points + all_images) // (all_images)

        if self.datamanager.dataparser.__class__.__name__ == "MushroomDataParser":
            # init mushroom eval lists
            metrics_dict_with_list = []
            metrics_dict_within_list = []
            points_with = []
            points_within = []
            colors_with = []
            colors_within = []
        else:
            # eval lists for other dataparsers
            metrics_dict_list = []
            points_eval = []
            colors_eval = []
        points_train = []
        colors_train = []

        # # compute eval metrics and generate eval point clouds
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                "[green]Evaluating all eval images...", total=num_eval
            )

            cameras = self.datamanager.eval_dataset.cameras  # type: ignore
            for image_idx, batch in enumerate(
                self.datamanager.cached_eval  # Undistorted images
            ):  # type: ignore
                camera = cameras[image_idx : image_idx + 1].to("cpu")
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, _ = self.model.get_image_metrics_and_images(
                    outputs, batch
                )
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (
                    num_rays / (time() - inner_start)
                ).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (
                    metrics_dict["num_rays_per_sec"] / (height * width)
                ).item()

                # get point cloud from each frame
                if "depth" in outputs and not self.config.skip_point_metrics:
                    depth = outputs["depth"]
                    rgb = outputs["rgb"]
                    indices = random.sample(range(pixels_per_frame), samples_per_frame)
                    c2w = torch.concatenate(
                        [
                            camera.camera_to_worlds,
                            torch.tensor([[[0, 0, 0, 1]]]).to(self.device),
                        ],
                        dim=1,
                    )
                    c2w = torch.matmul(
                        c2w,
                        torch.from_numpy(camera_utils.OPENGL_TO_OPENCV)
                        .float()
                        .to(depth.device),
                    )
                    fx, fy, cx, cy, img_size = (
                        camera.fx.item(),
                        camera.fy.item(),
                        camera.cx.item(),
                        camera.cy.item(),
                        (camera.width.item(), camera.height.item()),
                    )
                    if self._model.__class__.__name__ not in [
                        "DNSplatterModel",
                        "SplatfactoModel",
                    ]:
                        depth = depth / outputs["directions_norm"]

                    points, colors = camera_utils.get_colored_points_from_depth(
                        depths=depth,
                        rgbs=rgb,
                        c2w=c2w,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        img_size=img_size,
                        mask=indices,
                    )
                    points, colors = (
                        points.detach().cpu().numpy(),
                        colors.detach().cpu().numpy(),
                    )
                if (
                    self.datamanager.dataparser.__class__.__name__
                    == "MushroomDataParser"
                ):
                    seq_name = self.datamanager.eval_dataset.image_filenames[
                        batch["image_idx"]
                    ]
                    if "long_capture" in seq_name.parts[-3]:
                        metrics_dict_within_list.append(metrics_dict)
                        if not self.config.skip_point_metrics:
                            points_within.append(points)
                            colors_within.append(colors)
                    else:
                        metrics_dict_with_list.append(metrics_dict)
                        if not self.config.skip_point_metrics:
                            points_with.append(points)
                            colors_with.append(colors)
                else:
                    metrics_dict_list.append(metrics_dict)
                    if not self.config.skip_point_metrics:
                        points_eval.append(points)
                        colors_eval.append(colors)
                progress.advance(task)

        # save pointcloud from training images
        pd_metrics = {}
        if not self.config.skip_point_metrics:
            train_dataset = self.datamanager.train_dataset
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "[green]Extracting point cloud from train images...",
                    total=num_train,
                )
                for image_idx, _ in enumerate(train_dataset):
                    camera = train_dataset.cameras[image_idx : image_idx + 1].to(
                        self._model.device
                    )
                    outputs = self.model.get_outputs_for_camera(camera=camera)
                    rgb, depth = outputs["rgb"], outputs["depth"]
                    indices = random.sample(range(pixels_per_frame), samples_per_frame)
                    c2w = torch.concatenate(
                        [
                            camera.camera_to_worlds,
                            torch.tensor([[[0, 0, 0, 1]]]).to(self.device),
                        ],
                        dim=1,
                    )
                    c2w = torch.matmul(
                        c2w,
                        torch.from_numpy(camera_utils.OPENGL_TO_OPENCV)
                        .float()
                        .to(depth.device),
                    )
                    fx, fy, cx, cy, img_size = (
                        camera.fx.item(),
                        camera.fy.item(),
                        camera.cx.item(),
                        camera.cy.item(),
                        (camera.width.item(), camera.height.item()),
                    )
                    if self._model.__class__.__name__ not in [
                        "DNSplatterModel",
                        "SplatfactoModel",
                    ]:
                        depth = depth / outputs["directions_norm"]

                    points, colors = camera_utils.get_colored_points_from_depth(
                        depths=depth,
                        rgbs=rgb,
                        c2w=c2w,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        img_size=img_size,
                        mask=indices,
                    )
                    points, colors = (
                        points.detach().cpu().numpy(),
                        colors.detach().cpu().numpy(),
                    )
                    points_train.append(points)
                    colors_train.append(colors)
                    progress.advance(task)

            CONSOLE.print("[bold green]Computing point cloud metrics")
            pd_output_path = f"/{output_path}/final_renders"
            os.makedirs(os.getcwd() + f"{pd_output_path}", exist_ok=True)
            if self.datamanager.dataparser.__class__.__name__ == "MushroomDataParser":
                # load reference pcd for pointcloud comparison
                dataset_path = self.datamanager.dataparser_config.data
                ref_pcd_path = f"{dataset_path}/gt_pd.ply"
                if not os.path.exists(ref_pcd_path):
                    from dn_splatter.data.download_scripts.mushroom_download import (
                        download_mushroom,
                    )

                    download_mushroom(room_name=dataset_path.parts[-1], sequence="faro")
                ref_pcd = o3d.io.read_point_cloud(ref_pcd_path)
                transform_path = (
                    f"{dataset_path}/icp_{self.datamanager.dataparser_config.mode}.json"
                )
                initial_transformation = np.array(
                    json.load(open(transform_path))["gt_transformation"]
                ).reshape(4, 4)

                points_all = points_within + points_train
                colors_all = colors_within + colors_train
                points_all = np.vstack(points_all)
                colors_all = np.vstack(colors_all)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_all)
                pcd.colors = o3d.utility.Vector3dVector(colors_all)
                if self._model.__class__.__name__ not in [
                    "DNSplatterModel",
                    "SplatfactoModel",
                ]:
                    scale = self.datamanager.dataparser.scale_factor
                    transformation_matrix = self.datamanager.dataparser.transform_matrix
                    transformation_matrix = torch.cat(
                        [transformation_matrix, torch.tensor([0, 0, 0, 1]).unsqueeze(0)]
                    )
                    inverse_transformation = np.linalg.inv(transformation_matrix)

                    points = np.array(pcd.points) / scale
                    points = (
                        points @ inverse_transformation[:3, :3]
                        + inverse_transformation[:3, 3:4].T
                    )
                    pcd.points = o3d.utility.Vector3dVector(points)

                pcd = pcd.transform(initial_transformation)
                if output_path is not None:
                    o3d.io.write_point_cloud(
                        os.getcwd() + f"{pd_output_path}/pointcloud_within.ply", pcd
                    )

                acc, comp = self.pd_metrics(pcd, ref_pcd)
                pd_metrics.update(
                    {
                        "within_pd_acc": float(acc.item()),
                        "within_pd_comp": float(comp.item()),
                    }
                )

                points_all = points_with + points_train
                colors_all = colors_with + colors_train
                points_all = np.vstack(points_all)
                colors_all = np.vstack(colors_all)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_all)
                pcd.colors = o3d.utility.Vector3dVector(colors_all)
                if self._model.__class__.__name__ not in [
                    "DNSplatterModel",
                    "SplatfactoModel",
                ]:
                    scale = self.datamanager.dataparser.scale_factor
                    transformation_matrix = self.datamanager.dataparser.transform_matrix
                    transformation_matrix = torch.cat(
                        [transformation_matrix, torch.tensor([0, 0, 0, 1]).unsqueeze(0)]
                    )
                    inverse_transformation = np.linalg.inv(transformation_matrix)

                    points = np.array(pcd.points) / scale
                    points = (
                        points @ inverse_transformation[:3, :3]
                        + inverse_transformation[:3, 3:4].T
                    )
                    pcd.points = o3d.utility.Vector3dVector(points)

                pcd = pcd.transform(initial_transformation)
                if output_path is not None:
                    o3d.io.write_point_cloud(
                        os.getcwd() + f"{pd_output_path}/pointcloud_with.ply", pcd
                    )

                acc, comp = self.pd_metrics(pcd, ref_pcd)
                pd_metrics.update(
                    {
                        "with_pd_acc": float(acc.item()),
                        "with_pd_comp": float(comp.item()),
                    }
                )

            elif self.datamanager.dataparser.__class__.__name__ == "ReplicaDataparser":
                ref_pcd_path = self.config.datamanager.dataparser.data / (
                    self.config.datamanager.dataparser.sequence + "_mesh.ply"
                )  # load raplica mesh
                ref_mesh = trimesh.load_mesh(str(ref_pcd_path)).as_open3d
                ref_pcd = ref_mesh.sample_points_uniformly(
                    number_of_points=self.config.num_pd_points
                )
                points_all = points_eval + points_train
                colors_all = colors_eval + colors_train
                points_all = np.vstack(points_all)
                colors_all = np.vstack(colors_all)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_all)
                pcd.colors = o3d.utility.Vector3dVector(colors_all)

                if self._model.__class__.__name__ not in [
                    "DNSplatterModel",
                    "SplatfactoModel",
                ]:
                    scale = self.datamanager.dataparser.scale_factor
                    transformation_matrix = self.datamanager.dataparser.transform_matrix
                    transformation_matrix = torch.cat(
                        [transformation_matrix, torch.tensor([0, 0, 0, 1]).unsqueeze(0)]
                    )
                    inverse_transformation = np.linalg.inv(transformation_matrix)

                    points = np.array(pcd.points) / scale
                    points = (
                        points @ inverse_transformation[:3, :3]
                        + inverse_transformation[:3, 3:4].T
                    )
                    pcd.points = o3d.utility.Vector3dVector(points)

                if output_path is not None:
                    o3d.io.write_point_cloud(
                        os.getcwd() + f"{pd_output_path}/pointcloud.ply", pcd
                    )
                acc, comp = self.pd_metrics(pcd, ref_pcd)
                pd_metrics = {
                    "pd_acc": float(acc.item()),
                    "pd_comp": float(comp.item()),
                }
        # average the metrics list
        metrics_dict = {}

        if self.datamanager.dataparser.__class__.__name__ == "MushroomDataParser":
            for key in metrics_dict_within_list[0].keys():
                if get_std:
                    key_std, key_mean = torch.std_mean(
                        torch.tensor(
                            [
                                metrics_dict[key]
                                for metrics_dict in metrics_dict_within_list
                            ]
                        )
                    )
                    metrics_dict["within_" + key] = float(key_mean)
                    metrics_dict[f"within_{key}_std"] = float(key_std)
                else:
                    metrics_dict["within_" + key] = float(
                        torch.mean(
                            torch.tensor(
                                [
                                    metrics_dict["within_" + key]
                                    for metrics_dict in metrics_dict_within_list
                                ]
                            )
                        )
                    )
            for key in metrics_dict_with_list[0].keys():
                if get_std:
                    key_std, key_mean = torch.std_mean(
                        torch.tensor(
                            [
                                metrics_dict[key]
                                for metrics_dict in metrics_dict_with_list
                            ]
                        )
                    )
                    metrics_dict["with_" + key] = float(key_mean)
                    metrics_dict[f"with_{key}_std"] = float(key_std)
                else:
                    metrics_dict["with_" + key] = float(
                        torch.mean(
                            torch.tensor(
                                [
                                    metrics_dict[key]
                                    for metrics_dict in metrics_dict_with_list
                                ]
                            )
                        )
                    )
        else:
            for key in metrics_dict_list[0].keys():
                if get_std:
                    key_std, key_mean = torch.std_mean(
                        torch.tensor(
                            [metrics_dict[key] for metrics_dict in metrics_dict_list]
                        )
                    )
                    metrics_dict[key] = float(key_mean)
                    metrics_dict[f"{key}_std"] = float(key_std)
                else:
                    metrics_dict[key] = float(
                        torch.mean(
                            torch.tensor(
                                [
                                    metrics_dict[key]
                                    for metrics_dict in metrics_dict_list
                                ]
                            )
                        )
                    )
        metrics_dict.update(pd_metrics)
        self.train()

        ############################# secret evaluation ############################################
        camera_secret, data_secret = self.datamanager.next_train_idx(self.config_secret.secret_view_idx)
        model_outputs_secret = self.model(camera_secret) 
        
        # 1. compute masked lpips value
        mask_np = self.ip2p_ptd.mask
        # Convert mask to tensor and ensure it's the right shape/device
        mask_tensor = torch.from_numpy(mask_np).float()
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
        if mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
        mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
        mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

        # Prepare model output
        model_rgb_secret = (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

        # Apply mask to both images
        masked_model_rgb = model_rgb_secret * mask_tensor
        masked_ref_image = self.ref_image_tensor * mask_tensor

        # compute the masked lpips score of between the original ref image and the rendered secret image
        masked_lpips_score = self.lpips_loss_fn(
            masked_model_rgb,
            masked_ref_image
        ).squeeze()

        masked_secret_lpips_metrics = {
            "secret_masked_lpips": float(masked_lpips_score.detach().cpu().numpy()),
        }

        # update metrics_dict
        metrics_dict.update(masked_secret_lpips_metrics)

        # 2. compute miou metric
        rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2)
        save_image(rendered_image_secret, f"{output_path}/rendered_image.png")

        original_image_logits = self.lseg_model.decode_feature(self.original_image_sem_feature)
        original_semantic = self.lseg_model.visualize_sem(original_image_logits) # (c, h, w)
        save_image(original_semantic, f"{output_path}/original_semantic.png")
        original_pred = original_image_logits.argmax(dim=1, keepdim=True)
        original_pred = original_pred.clamp(max=7) + 1

        # [1, 512, 256, 256]
        rendered_image_sem_feature = self.lseg_model.get_image_features(rendered_image_secret.to(self.config_secret.device))
        rendered_image_logits = self.lseg_model.decode_feature(rendered_image_sem_feature)
        rendered_semantic = self.lseg_model.visualize_sem(rendered_image_logits) # (c, h, w)
        save_image(rendered_semantic, f"{output_path}/rendered_semantic.png")

        rendered_pred = rendered_image_logits.argmax(dim=1, keepdim=True)
        rendered_pred = rendered_pred.clamp(max=7) + 1

        self.miou.update(rendered_pred.to(self.config_secret.device), original_pred.to(self.config_secret.device))

        miou_metrics = {
            "miou": self.miou.compute().mean().item()
        }
        metrics_dict.update(miou_metrics)

        # 3. compute clip score
        clip, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        model = clip.eval().to(self.config_secret.device)
        text_features = model.encode_text(clip_tokenizer([self.config_secret.prompt_2]).to(self.config_secret.device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

        clip_img = rendered_image_secret.squeeze(0)
        clip_img = transforms.ToPILImage()(clip_img).convert("RGB")
        clip_img = clip_preprocess(clip_img).unsqueeze(0).to(self.config_secret.device)
        with torch.no_grad():
            image_features = model.encode_image(clip_img)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        clip_score = torch.nn.functional.cosine_similarity(image_features, text_features).item()

        clip_metrics = {
            "clip_score": clip_score
        }
        metrics_dict.update(clip_metrics)
        ############################# secret evaluation ############################################

        # render images
        if output_path is not None:
            # render gs model images
            CONSOLE.print("[bold green]Rendering output images")
            if self._model.__class__.__name__ in ["DNSplatterModel", "SplatfactoModel"]:
                render_output_path = f"/{output_path}/final_renders"
                train_cache = self.datamanager.cached_train
                eval_cache = self.datamanager.cached_eval
                train_dataset = self.datamanager.train_dataset
                eval_dataset = self.datamanager.eval_dataset
                model = self._model
                gs_render_dataset_images(
                    train_cache=train_cache,
                    eval_cache=eval_cache,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    model=model,
                    render_output_path=render_output_path,
                    mushroom=(
                        True
                        if self.datamanager.dataparser.__class__.__name__
                        == "MushroomDataParser"
                        else False
                    ),
                    save_train_images=self.config.save_train_images,
                )
            else:
                # render other models
                print("Rendering for ", self._model.__class__.__name__)
                render_output_path = f"/{output_path}/final_renders"
                train_dataset = self.datamanager.train_dataset
                eval_dataset = self.datamanager.eval_dataset
                model = self._model
                train_dataloader = FixedIndicesEvalDataloader(
                    input_dataset=train_dataset,
                    device=self.datamanager.device,
                    num_workers=self.datamanager.world_size * 4,
                )
                eval_dataloader = FixedIndicesEvalDataloader(
                    input_dataset=eval_dataset,
                    device=self.datamanager.device,
                    num_workers=self.datamanager.world_size * 4,
                )
                ns_render_dataset_images(
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_dataloader,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    model=model,
                    render_output_path=render_output_path,
                    mushroom=(
                        True
                        if self.datamanager.dataparser.__class__.__name__
                        == "MushroomDataParser"
                        else False
                    ),
                    save_train_images=self.config.save_train_images,
                )

        # compare rendered depth with faro depth for mushroom dataset
        if self.datamanager.dataparser.__class__.__name__ == "MushroomDataParser":
            if output_path is not None:
                faro_depth_path = (
                    self.datamanager.dataparser_config.data
                    / self.datamanager.dataparser_config.mode
                )
                faro_metrics = depth_eval_faro(output_path, faro_depth_path)
                metrics_dict.update(faro_metrics)

        return metrics_dict
    

    ### original eval function
    # @profiler.time_function
    # def get_average_eval_image_metrics(
    #     self,
    #     step: Optional[int] = None,
    #     output_path: Optional[Path] = None,
    #     get_std: bool = False,
    # ):
    #     """Iterate over all the images in the eval dataset and get the average.

    #     Args:
    #         step: current training step
    #         output_path: optional path to save rendered images to
    #         get_std: Set True if you want to return std with the mean metric.

    #     Returns:
    #         metrics_dict: dictionary of metrics
    #     """
    #     self.eval()
    #     assert isinstance(
    #         self.datamanager,
    #         (VanillaDataManager, ParallelDataManager, FullImageDatamanager),
    #     )
    #     num_eval = len(self.datamanager.fixed_indices_eval_dataloader)
    #     num_train = len(self.datamanager.train_dataset)  # type: ignore
    #     all_images = num_train + num_eval

    #     if not self.config.skip_point_metrics:
    #         pixels_per_frame = int(
    #             self.datamanager.train_dataset.cameras[0].width
    #             * self.datamanager.train_dataset.cameras[0].height
    #         )
    #         samples_per_frame = (self.config.num_pd_points + all_images) // (all_images)

    #     if self.datamanager.dataparser.__class__.__name__ == "MushroomDataParser":
    #         # init mushroom eval lists
    #         metrics_dict_with_list = []
    #         metrics_dict_within_list = []
    #         points_with = []
    #         points_within = []
    #         colors_with = []
    #         colors_within = []
    #     else:
    #         # eval lists for other dataparsers
    #         metrics_dict_list = []
    #         points_eval = []
    #         colors_eval = []
    #     points_train = []
    #     colors_train = []

    #     # # compute eval metrics and generate eval point clouds
    #     with Progress(
    #         TextColumn("[progress.description]{task.description}"),
    #         BarColumn(),
    #         TimeElapsedColumn(),
    #         MofNCompleteColumn(),
    #         transient=True,
    #     ) as progress:
    #         task = progress.add_task(
    #             "[green]Evaluating all eval images...", total=num_eval
    #         )

    #         cameras = self.datamanager.eval_dataset.cameras  # type: ignore
    #         for image_idx, batch in enumerate(
    #             self.datamanager.cached_eval  # Undistorted images
    #         ):  # type: ignore
    #             camera = cameras[image_idx : image_idx + 1].to("cpu")
    #             # time this the following line
    #             inner_start = time()
    #             outputs = self.model.get_outputs_for_camera(camera=camera)
    #             height, width = camera.height, camera.width
    #             num_rays = height * width
    #             metrics_dict, _ = self.model.get_image_metrics_and_images(
    #                 outputs, batch
    #             )
    #             assert "num_rays_per_sec" not in metrics_dict
    #             metrics_dict["num_rays_per_sec"] = (
    #                 num_rays / (time() - inner_start)
    #             ).item()
    #             fps_str = "fps"
    #             assert fps_str not in metrics_dict
    #             metrics_dict[fps_str] = (
    #                 metrics_dict["num_rays_per_sec"] / (height * width)
    #             ).item()

    #             # get point cloud from each frame
    #             if "depth" in outputs and not self.config.skip_point_metrics:
    #                 depth = outputs["depth"]
    #                 rgb = outputs["rgb"]
    #                 indices = random.sample(range(pixels_per_frame), samples_per_frame)
    #                 c2w = torch.concatenate(
    #                     [
    #                         camera.camera_to_worlds,
    #                         torch.tensor([[[0, 0, 0, 1]]]).to(self.device),
    #                     ],
    #                     dim=1,
    #                 )
    #                 c2w = torch.matmul(
    #                     c2w,
    #                     torch.from_numpy(camera_utils.OPENGL_TO_OPENCV)
    #                     .float()
    #                     .to(depth.device),
    #                 )
    #                 fx, fy, cx, cy, img_size = (
    #                     camera.fx.item(),
    #                     camera.fy.item(),
    #                     camera.cx.item(),
    #                     camera.cy.item(),
    #                     (camera.width.item(), camera.height.item()),
    #                 )
    #                 if self._model.__class__.__name__ not in [
    #                     "DNSplatterModel",
    #                     "SplatfactoModel",
    #                 ]:
    #                     depth = depth / outputs["directions_norm"]

    #                 points, colors = camera_utils.get_colored_points_from_depth(
    #                     depths=depth,
    #                     rgbs=rgb,
    #                     c2w=c2w,
    #                     fx=fx,
    #                     fy=fy,
    #                     cx=cx,
    #                     cy=cy,
    #                     img_size=img_size,
    #                     mask=indices,
    #                 )
    #                 points, colors = (
    #                     points.detach().cpu().numpy(),
    #                     colors.detach().cpu().numpy(),
    #                 )
    #             if (
    #                 self.datamanager.dataparser.__class__.__name__
    #                 == "MushroomDataParser"
    #             ):
    #                 seq_name = self.datamanager.eval_dataset.image_filenames[
    #                     batch["image_idx"]
    #                 ]
    #                 if "long_capture" in seq_name.parts[-3]:
    #                     metrics_dict_within_list.append(metrics_dict)
    #                     if not self.config.skip_point_metrics:
    #                         points_within.append(points)
    #                         colors_within.append(colors)
    #                 else:
    #                     metrics_dict_with_list.append(metrics_dict)
    #                     if not self.config.skip_point_metrics:
    #                         points_with.append(points)
    #                         colors_with.append(colors)
    #             else:
    #                 metrics_dict_list.append(metrics_dict)
    #                 if not self.config.skip_point_metrics:
    #                     points_eval.append(points)
    #                     colors_eval.append(colors)
    #             progress.advance(task)

    #     # save pointcloud from training images
    #     pd_metrics = {}
    #     if not self.config.skip_point_metrics:
    #         train_dataset = self.datamanager.train_dataset
    #         with Progress(
    #             TextColumn("[progress.description]{task.description}"),
    #             BarColumn(),
    #             TimeElapsedColumn(),
    #             MofNCompleteColumn(),
    #             transient=True,
    #         ) as progress:
    #             task = progress.add_task(
    #                 "[green]Extracting point cloud from train images...",
    #                 total=num_train,
    #             )
    #             for image_idx, _ in enumerate(train_dataset):
    #                 camera = train_dataset.cameras[image_idx : image_idx + 1].to(
    #                     self._model.device
    #                 )
    #                 outputs = self.model.get_outputs_for_camera(camera=camera)
    #                 rgb, depth = outputs["rgb"], outputs["depth"]
    #                 indices = random.sample(range(pixels_per_frame), samples_per_frame)
    #                 c2w = torch.concatenate(
    #                     [
    #                         camera.camera_to_worlds,
    #                         torch.tensor([[[0, 0, 0, 1]]]).to(self.device),
    #                     ],
    #                     dim=1,
    #                 )
    #                 c2w = torch.matmul(
    #                     c2w,
    #                     torch.from_numpy(camera_utils.OPENGL_TO_OPENCV)
    #                     .float()
    #                     .to(depth.device),
    #                 )
    #                 fx, fy, cx, cy, img_size = (
    #                     camera.fx.item(),
    #                     camera.fy.item(),
    #                     camera.cx.item(),
    #                     camera.cy.item(),
    #                     (camera.width.item(), camera.height.item()),
    #                 )
    #                 if self._model.__class__.__name__ not in [
    #                     "DNSplatterModel",
    #                     "SplatfactoModel",
    #                 ]:
    #                     depth = depth / outputs["directions_norm"]

    #                 points, colors = camera_utils.get_colored_points_from_depth(
    #                     depths=depth,
    #                     rgbs=rgb,
    #                     c2w=c2w,
    #                     fx=fx,
    #                     fy=fy,
    #                     cx=cx,
    #                     cy=cy,
    #                     img_size=img_size,
    #                     mask=indices,
    #                 )
    #                 points, colors = (
    #                     points.detach().cpu().numpy(),
    #                     colors.detach().cpu().numpy(),
    #                 )
    #                 points_train.append(points)
    #                 colors_train.append(colors)
    #                 progress.advance(task)

    #         CONSOLE.print("[bold green]Computing point cloud metrics")
    #         pd_output_path = f"/{output_path}/final_renders"
    #         os.makedirs(os.getcwd() + f"{pd_output_path}", exist_ok=True)
    #         if self.datamanager.dataparser.__class__.__name__ == "MushroomDataParser":
    #             # load reference pcd for pointcloud comparison
    #             dataset_path = self.datamanager.dataparser_config.data
    #             ref_pcd_path = f"{dataset_path}/gt_pd.ply"
    #             if not os.path.exists(ref_pcd_path):
    #                 from dn_splatter.data.download_scripts.mushroom_download import (
    #                     download_mushroom,
    #                 )

    #                 download_mushroom(room_name=dataset_path.parts[-1], sequence="faro")
    #             ref_pcd = o3d.io.read_point_cloud(ref_pcd_path)
    #             transform_path = (
    #                 f"{dataset_path}/icp_{self.datamanager.dataparser_config.mode}.json"
    #             )
    #             initial_transformation = np.array(
    #                 json.load(open(transform_path))["gt_transformation"]
    #             ).reshape(4, 4)

    #             points_all = points_within + points_train
    #             colors_all = colors_within + colors_train
    #             points_all = np.vstack(points_all)
    #             colors_all = np.vstack(colors_all)
    #             pcd = o3d.geometry.PointCloud()
    #             pcd.points = o3d.utility.Vector3dVector(points_all)
    #             pcd.colors = o3d.utility.Vector3dVector(colors_all)
    #             if self._model.__class__.__name__ not in [
    #                 "DNSplatterModel",
    #                 "SplatfactoModel",
    #             ]:
    #                 scale = self.datamanager.dataparser.scale_factor
    #                 transformation_matrix = self.datamanager.dataparser.transform_matrix
    #                 transformation_matrix = torch.cat(
    #                     [transformation_matrix, torch.tensor([0, 0, 0, 1]).unsqueeze(0)]
    #                 )
    #                 inverse_transformation = np.linalg.inv(transformation_matrix)

    #                 points = np.array(pcd.points) / scale
    #                 points = (
    #                     points @ inverse_transformation[:3, :3]
    #                     + inverse_transformation[:3, 3:4].T
    #                 )
    #                 pcd.points = o3d.utility.Vector3dVector(points)

    #             pcd = pcd.transform(initial_transformation)
    #             if output_path is not None:
    #                 o3d.io.write_point_cloud(
    #                     os.getcwd() + f"{pd_output_path}/pointcloud_within.ply", pcd
    #                 )

    #             acc, comp = self.pd_metrics(pcd, ref_pcd)
    #             pd_metrics.update(
    #                 {
    #                     "within_pd_acc": float(acc.item()),
    #                     "within_pd_comp": float(comp.item()),
    #                 }
    #             )

    #             points_all = points_with + points_train
    #             colors_all = colors_with + colors_train
    #             points_all = np.vstack(points_all)
    #             colors_all = np.vstack(colors_all)
    #             pcd = o3d.geometry.PointCloud()
    #             pcd.points = o3d.utility.Vector3dVector(points_all)
    #             pcd.colors = o3d.utility.Vector3dVector(colors_all)
    #             if self._model.__class__.__name__ not in [
    #                 "DNSplatterModel",
    #                 "SplatfactoModel",
    #             ]:
    #                 scale = self.datamanager.dataparser.scale_factor
    #                 transformation_matrix = self.datamanager.dataparser.transform_matrix
    #                 transformation_matrix = torch.cat(
    #                     [transformation_matrix, torch.tensor([0, 0, 0, 1]).unsqueeze(0)]
    #                 )
    #                 inverse_transformation = np.linalg.inv(transformation_matrix)

    #                 points = np.array(pcd.points) / scale
    #                 points = (
    #                     points @ inverse_transformation[:3, :3]
    #                     + inverse_transformation[:3, 3:4].T
    #                 )
    #                 pcd.points = o3d.utility.Vector3dVector(points)

    #             pcd = pcd.transform(initial_transformation)
    #             if output_path is not None:
    #                 o3d.io.write_point_cloud(
    #                     os.getcwd() + f"{pd_output_path}/pointcloud_with.ply", pcd
    #                 )

    #             acc, comp = self.pd_metrics(pcd, ref_pcd)
    #             pd_metrics.update(
    #                 {
    #                     "with_pd_acc": float(acc.item()),
    #                     "with_pd_comp": float(comp.item()),
    #                 }
    #             )

    #         elif self.datamanager.dataparser.__class__.__name__ == "ReplicaDataparser":
    #             ref_pcd_path = self.config.datamanager.dataparser.data / (
    #                 self.config.datamanager.dataparser.sequence + "_mesh.ply"
    #             )  # load raplica mesh
    #             ref_mesh = trimesh.load_mesh(str(ref_pcd_path)).as_open3d
    #             ref_pcd = ref_mesh.sample_points_uniformly(
    #                 number_of_points=self.config.num_pd_points
    #             )
    #             points_all = points_eval + points_train
    #             colors_all = colors_eval + colors_train
    #             points_all = np.vstack(points_all)
    #             colors_all = np.vstack(colors_all)
    #             pcd = o3d.geometry.PointCloud()
    #             pcd.points = o3d.utility.Vector3dVector(points_all)
    #             pcd.colors = o3d.utility.Vector3dVector(colors_all)

    #             if self._model.__class__.__name__ not in [
    #                 "DNSplatterModel",
    #                 "SplatfactoModel",
    #             ]:
    #                 scale = self.datamanager.dataparser.scale_factor
    #                 transformation_matrix = self.datamanager.dataparser.transform_matrix
    #                 transformation_matrix = torch.cat(
    #                     [transformation_matrix, torch.tensor([0, 0, 0, 1]).unsqueeze(0)]
    #                 )
    #                 inverse_transformation = np.linalg.inv(transformation_matrix)

    #                 points = np.array(pcd.points) / scale
    #                 points = (
    #                     points @ inverse_transformation[:3, :3]
    #                     + inverse_transformation[:3, 3:4].T
    #                 )
    #                 pcd.points = o3d.utility.Vector3dVector(points)

    #             if output_path is not None:
    #                 o3d.io.write_point_cloud(
    #                     os.getcwd() + f"{pd_output_path}/pointcloud.ply", pcd
    #                 )
    #             acc, comp = self.pd_metrics(pcd, ref_pcd)
    #             pd_metrics = {
    #                 "pd_acc": float(acc.item()),
    #                 "pd_comp": float(comp.item()),
    #             }
    #     # average the metrics list
    #     metrics_dict = {}

    #     if self.datamanager.dataparser.__class__.__name__ == "MushroomDataParser":
    #         for key in metrics_dict_within_list[0].keys():
    #             if get_std:
    #                 key_std, key_mean = torch.std_mean(
    #                     torch.tensor(
    #                         [
    #                             metrics_dict[key]
    #                             for metrics_dict in metrics_dict_within_list
    #                         ]
    #                     )
    #                 )
    #                 metrics_dict["within_" + key] = float(key_mean)
    #                 metrics_dict[f"within_{key}_std"] = float(key_std)
    #             else:
    #                 metrics_dict["within_" + key] = float(
    #                     torch.mean(
    #                         torch.tensor(
    #                             [
    #                                 metrics_dict["within_" + key]
    #                                 for metrics_dict in metrics_dict_within_list
    #                             ]
    #                         )
    #                     )
    #                 )
    #         for key in metrics_dict_with_list[0].keys():
    #             if get_std:
    #                 key_std, key_mean = torch.std_mean(
    #                     torch.tensor(
    #                         [
    #                             metrics_dict[key]
    #                             for metrics_dict in metrics_dict_with_list
    #                         ]
    #                     )
    #                 )
    #                 metrics_dict["with_" + key] = float(key_mean)
    #                 metrics_dict[f"with_{key}_std"] = float(key_std)
    #             else:
    #                 metrics_dict["with_" + key] = float(
    #                     torch.mean(
    #                         torch.tensor(
    #                             [
    #                                 metrics_dict[key]
    #                                 for metrics_dict in metrics_dict_with_list
    #                             ]
    #                         )
    #                     )
    #                 )
    #     else:
    #         for key in metrics_dict_list[0].keys():
    #             if get_std:
    #                 key_std, key_mean = torch.std_mean(
    #                     torch.tensor(
    #                         [metrics_dict[key] for metrics_dict in metrics_dict_list]
    #                     )
    #                 )
    #                 metrics_dict[key] = float(key_mean)
    #                 metrics_dict[f"{key}_std"] = float(key_std)
    #             else:
    #                 metrics_dict[key] = float(
    #                     torch.mean(
    #                         torch.tensor(
    #                             [
    #                                 metrics_dict[key]
    #                                 for metrics_dict in metrics_dict_list
    #                             ]
    #                         )
    #                     )
    #                 )
    #     metrics_dict.update(pd_metrics)
    #     self.train()

    #     # render images
    #     if output_path is not None:
    #         # render gs model images
    #         CONSOLE.print("[bold green]Rendering output images")
    #         if self._model.__class__.__name__ in ["DNSplatterModel", "SplatfactoModel"]:
    #             render_output_path = f"/{output_path}/final_renders"
    #             train_cache = self.datamanager.cached_train
    #             eval_cache = self.datamanager.cached_eval
    #             train_dataset = self.datamanager.train_dataset
    #             eval_dataset = self.datamanager.eval_dataset
    #             model = self._model
    #             gs_render_dataset_images(
    #                 train_cache=train_cache,
    #                 eval_cache=eval_cache,
    #                 train_dataset=train_dataset,
    #                 eval_dataset=eval_dataset,
    #                 model=model,
    #                 render_output_path=render_output_path,
    #                 mushroom=(
    #                     True
    #                     if self.datamanager.dataparser.__class__.__name__
    #                     == "MushroomDataParser"
    #                     else False
    #                 ),
    #                 save_train_images=self.config.save_train_images,
    #             )
    #         else:
    #             # render other models
    #             print("Rendering for ", self._model.__class__.__name__)
    #             render_output_path = f"/{output_path}/final_renders"
    #             train_dataset = self.datamanager.train_dataset
    #             eval_dataset = self.datamanager.eval_dataset
    #             model = self._model
    #             train_dataloader = FixedIndicesEvalDataloader(
    #                 input_dataset=train_dataset,
    #                 device=self.datamanager.device,
    #                 num_workers=self.datamanager.world_size * 4,
    #             )
    #             eval_dataloader = FixedIndicesEvalDataloader(
    #                 input_dataset=eval_dataset,
    #                 device=self.datamanager.device,
    #                 num_workers=self.datamanager.world_size * 4,
    #             )
    #             ns_render_dataset_images(
    #                 train_dataloader=train_dataloader,
    #                 eval_dataloader=eval_dataloader,
    #                 train_dataset=train_dataset,
    #                 eval_dataset=eval_dataset,
    #                 model=model,
    #                 render_output_path=render_output_path,
    #                 mushroom=(
    #                     True
    #                     if self.datamanager.dataparser.__class__.__name__
    #                     == "MushroomDataParser"
    #                     else False
    #                 ),
    #                 save_train_images=self.config.save_train_images,
    #             )

    #     # compare rendered depth with faro depth for mushroom dataset
    #     if self.datamanager.dataparser.__class__.__name__ == "MushroomDataParser":
    #         if output_path is not None:
    #             faro_depth_path = (
    #                 self.datamanager.dataparser_config.data
    #                 / self.datamanager.dataparser_config.mode
    #             )
    #             faro_metrics = depth_eval_faro(output_path, faro_depth_path)
    #             metrics_dict.update(faro_metrics)

    #     return metrics_dict


    