"""
Datamanager that processes optional depth, semantic and normal data.
"""

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch
import torchvision.transforms.functional as TF

import cv2
import numpy as np

from dn_splatter.data.dn_dataset import GDataset
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.datasets.base_dataset import InputDataset


@dataclass
class DNSplatterManagerConfig(FullImageDatamanagerConfig):
    """DataManager Config"""

    _target: Type = field(default_factory=lambda: DNSplatterDataManager)

    camera_res_scale_factor: float = 1.0
    """Rescale cameras"""


class DNSplatterDataManager(FullImageDatamanager):
    """DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: DNSplatterManagerConfig
    train_dataset: GDataset
    eval_dataset: GDataset

    def __init__(
        self,
        config: DNSplatterManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )

        ######################################################
        # Secret view updating                              
        ######################################################
        # print("\n meta data semantic filenames: \n", self.train_dataparser_outputs.metadata["semantic_filenames"])
        # from igs2gs_datamanager.py
        # add depth into the cache
        depth_fnames = self.train_dataparser_outputs.metadata.get("depth_filenames", None)
        if depth_fnames is not None:
            # choose whether to keep CPU or move to GPU based on your cache_images setting
            to_device = (lambda x: x.to(self.device)) if self.config.cache_images == "gpu" else (lambda x: x)
            for sample, depth_path in zip(self.cached_train, depth_fnames):
                depth_np = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
                depth_t  = torch.from_numpy(depth_np)[None, ...]
                sample["depth"] = to_device(depth_t)

        # semantic adaptation
        semantic_fnames = self.train_dataparser_outputs.metadata.get("semantic_filenames", None)
        if semantic_fnames is not None:
            # choose whether to keep CPU or move to GPU based on your cache_images setting
            to_device = (lambda x: x.to(self.device)) if self.config.cache_images == "gpu" else (lambda x: x)
            for sample, semantic_path in zip(self.cached_train, semantic_fnames):
                semantic_np = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
                semantic_t  = torch.from_numpy(semantic_np)[None, ...]
                sample["semantic"] = to_device(semantic_t)

        # cache original training images for ip2p
        self.original_cached_train = deepcopy(self.cached_train)
        self.original_cached_eval = deepcopy(self.cached_eval)
        ######################################################
        

        metadata = self.train_dataparser_outputs.metadata
        self.load_depths = (
            True
            if ("depth_filenames" in metadata)
            or ("sensor_depth_filenames" in metadata)
            or ("mono_depth_filenames") in metadata
            else False
        )

        self.load_semantics = True if ("semantic_filenames" in metadata) else False

        self.load_normals = True if ("normal_filenames" in metadata) else False
        self.load_confidence = True if ("confidence_filenames" in metadata) else False
        self.image_idx = 0

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return GDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        return GDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(
                split=self.test_split
            ),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch"""

        # Don't randomly sample train images (keep t-1, t, t+1 ordering).
        self.image_idx = self.train_unseen_cameras.pop(0)
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
        data = deepcopy(self.cached_train[self.image_idx])
        data["image"] = data["image"].to(self.device)

        if "mask" in data:
            data["mask"] = data["mask"].to(self.device)
            if data["mask"].dim() == 2:
                data["mask"] = data["mask"][..., None]

        if self.load_depths:
            if "sensor_depth" in data:
                data["sensor_depth"] = data["sensor_depth"].to(self.device)
                if data["sensor_depth"].shape != data["image"].shape:
                    data["sensor_depth"] = TF.resize(
                        data["sensor_depth"].permute(2, 0, 1),
                        data["image"].shape[:2],
                        antialias=None,
                    ).permute(1, 2, 0)
            if "mono_depth" in data:
                data["mono_depth"] = data["mono_depth"].to(self.device)
                if data["mono_depth"].shape != data["image"].shape:
                    data["mono_depth"] = TF.resize(
                        data["mono_depth"].permute(2, 0, 1),
                        data["image"].shape[:2],
                        antialias=None,
                    ).permute(1, 2, 0)

        if self.load_normals:
            assert "normal" in data
            data["normal"] = data["normal"].to(self.device)
            if data["normal"].shape != data["image"].shape:
                data["normal"] = TF.resize(
                    data["normal"].permute(2, 0, 1),
                    data["image"].shape[:2],
                    antialias=None,
                ).permute(1, 2, 0)
                
        # load semantics
        if self.load_semantics:
            assert "semantic" in data
            data["semantic"] = data["semantic"].to(self.device)
            if data["semantic"].shape != data["image"].shape:
                data["semantic"] = data["semantic"].squeeze(0)
                data["semantic"] = TF.resize(
                    data["semantic"].permute(2, 0, 1),
                    data["image"].shape[:2],
                    antialias=None,
                ).permute(1, 2, 0)
                data["semantic"] = data["semantic"].unsqueeze(0)
        
        if self.load_confidence:
            assert "confidence" in data
            data["confidence"] = data["confidence"].to(self.device)
            if data["confidence"].shape != data["image"].shape:
                data["confidence"] = TF.resize(
                    data["confidence"].permute(2, 0, 1),
                    data["image"].shape[:2],
                    antialias=None,
                ).permute(1, 2, 0)
        assert (
            len(self.train_dataset.cameras.shape) == 1
        ), "Assumes single batch dimension"
        camera = self.train_dataset.cameras[self.image_idx : self.image_idx + 1].to(
            self.device
        )
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = self.image_idx
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        image_idx = self.eval_unseen_cameras[
            random.randint(0, len(self.eval_unseen_cameras) - 1)
        ]

        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        if "mask" in data:
            data["mask"] = data["mask"].to(self.device)
            if data["mask"].dim() == 2:
                data["mask"] = data["mask"][..., None]
        if self.load_depths:
            if "sensor_depth" in data:
                data["sensor_depth"] = data["sensor_depth"].to(self.device)
                if data["sensor_depth"].shape != data["image"].shape:
                    data["sensor_depth"] = TF.resize(
                        data["sensor_depth"].permute(2, 0, 1),
                        data["image"].shape[:2],
                        antialias=None,
                    ).permute(1, 2, 0)
            if "mono_depth" in data:
                data["mono_depth"] = data["mono_depth"].to(self.device)
                if data["mono_depth"].shape != data["image"].shape:
                    data["mono_depth"] = TF.resize(
                        data["mono_depth"].permute(2, 0, 1),
                        data["image"].shape[:2],
                        antialias=None,
                    ).permute(1, 2, 0)
        if self.load_normals:
            assert "normal" in data
            data["normal"] = data["normal"].to(self.device)
            if data["normal"].shape != data["image"].shape:
                data["normal"] = TF.resize(
                    data["normal"].permute(2, 0, 1),
                    data["image"].shape[:2],
                    antialias=None,
                ).permute(1, 2, 0)
        # load semantics
        if self.load_semantics:
            assert "semantic" in data
            data["semantic"] = data["semantic"].to(self.device)
            if data["semantic"].shape != data["image"].shape:
                data["semantic"] = TF.resize(
                    data["semantic"].permute(2, 0, 1),
                    data["image"].shape[:2],
                    antialias=None,
                ).permute(1, 2, 0)
        if self.load_confidence:
            assert "confidence" in data
            data["confidence"] = data["confidence"].to(self.device)
            if data["confidence"].shape != data["image"].shape:
                data["confidence"] = TF.resize(
                    data["confidence"].permute(2, 0, 1),
                    data["image"].shape[:2],
                    antialias=None,
                ).permute(1, 2, 0)
        assert (
            len(self.eval_dataset.cameras.shape) == 1
        ), "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next eval image"""

        image_idx = self.eval_unseen_cameras[
            random.randint(0, len(self.eval_unseen_cameras) - 1)
        ]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        assert (
            len(self.eval_dataset.cameras.shape) == 1
        ), "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data

    # from igs2gs_datamanager.py
    def next_train_idx(self, idx: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        data = deepcopy(self.cached_train[idx])
        data["image"] = data["image"].to(self.device)
        # add depth into the cache
        data["depth"] = data["depth"].to(self.device)
        if "mask" in data:
            data["mask"] = data["mask"].to(self.device)
        if self.load_depths:
            if "sensor_depth" in data:
                data["sensor_depth"] = data["sensor_depth"].to(self.device)
            if "mono_depth" in data:
                data["mono_depth"] = data["mono_depth"].to(self.device)
        if self.load_normals:
            assert "normal" in data
            data["normal"] = data["normal"].to(self.device)
        # load semantics
        if self.load_semantics:
            assert "semantic" in data
            data["semantic"] = data["semantic"].to(self.device)
        if self.load_confidence:
            assert "confidence" in data
            data["confidence"] = data["confidence"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[idx : idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = idx
        return camera, data

    # dataset downsampling
    def downsample_dataset(self, scale_factor: float, secret_idx: int = None) -> int:
        """Downsample the dataset by a given scale factor while preserving a specific index.
        
        Args:
            scale_factor: Factor by which to downsample the dataset. 
                        E.g., 2.0 means keep 1/2 of the data, 
                        4.0 means keep 1/4 of the data.
            secret_idx: Index in the original dataset that must be preserved.
                    If None, no specific index is preserved.
        
        Returns:
            New index of the secret data in the downsampled dataset.
            Returns -1 if secret_idx was None.
        """
        if scale_factor <= 1.0:
            # No downsampling needed
            return secret_idx if secret_idx is not None else -1
        
        # Calculate the number of samples to keep
        original_train_size = len(self.train_dataset)
        original_eval_size = len(self.eval_dataset)
        
        new_train_size = max(1, int(original_train_size / scale_factor))
        new_eval_size = max(1, int(original_eval_size / scale_factor))
        
        # Generate indices for subsampling (evenly spaced to maintain temporal/spatial coverage)
        train_indices = np.linspace(0, original_train_size - 1, new_train_size, dtype=int).tolist()
        eval_indices = np.linspace(0, original_eval_size - 1, new_eval_size, dtype=int).tolist()
        
        # Handle secret_idx preservation for training set
        new_secret_idx = -1
        if secret_idx is not None and 0 <= secret_idx < original_train_size:
            # Check if secret_idx is already in the sampled indices
            if secret_idx not in train_indices:
                # Find the best position to insert secret_idx to maintain ordering
                insert_pos = 0
                for i, idx in enumerate(train_indices):
                    if idx < secret_idx:
                        insert_pos = i + 1
                    else:
                        break
                
                # Insert the secret_idx at the appropriate position
                train_indices.insert(insert_pos, secret_idx)
                new_secret_idx = insert_pos
            else:
                # Secret_idx is already in the list, find its position
                new_secret_idx = train_indices.index(secret_idx)
        
        # Sort indices to ensure proper ordering
        train_indices = sorted(train_indices)
        
        # Update new_secret_idx after sorting if it was added
        if secret_idx is not None and 0 <= secret_idx < original_train_size:
            new_secret_idx = train_indices.index(secret_idx)
        
        # Downsample training dataset
        if hasattr(self, 'cached_train'):
            self.cached_train = [self.cached_train[i] for i in train_indices]
        
        if hasattr(self, 'original_cached_train'):
            self.original_cached_train = [self.original_cached_train[i] for i in train_indices]
        
        # Downsample training cameras
        if hasattr(self.train_dataset, 'cameras'):
            # Convert list to tensor for proper indexing
            indices_tensor = torch.tensor(train_indices, dtype=torch.long)
            self.train_dataset.cameras = self.train_dataset.cameras[indices_tensor]
        
        # Downsample other training dataset attributes if they exist
        if hasattr(self.train_dataset, 'metadata'):
            for key, value in self.train_dataset.metadata.items():
                if isinstance(value, (list, np.ndarray)) and len(value) == original_train_size:
                    if isinstance(value, list):
                        self.train_dataset.metadata[key] = [value[i] for i in train_indices]
                    else:
                        self.train_dataset.metadata[key] = value[train_indices]
        
        # Update train dataset length
        if hasattr(self.train_dataset, '_dataparser_outputs'):
            self.train_dataset._dataparser_outputs.dataparser_scale = scale_factor
            
            # Update image filenames in dataparser outputs if they exist
            if hasattr(self.train_dataset._dataparser_outputs, 'image_filenames'):
                original_filenames = self.train_dataset._dataparser_outputs.image_filenames
                if original_filenames is not None:
                    self.train_dataset._dataparser_outputs.image_filenames = [
                        original_filenames[i] for i in train_indices
                    ]
        
        # Downsample evaluation dataset
        if hasattr(self, 'cached_eval'):
            self.cached_eval = [self.cached_eval[i] for i in eval_indices]
        
        if hasattr(self, 'original_cached_eval'):
            self.original_cached_eval = [self.original_cached_eval[i] for i in eval_indices]
        
        # Downsample evaluation cameras
        if hasattr(self.eval_dataset, 'cameras'):
            # Convert list to tensor for proper indexing
            eval_indices_tensor = torch.tensor(eval_indices, dtype=torch.long)
            self.eval_dataset.cameras = self.eval_dataset.cameras[eval_indices_tensor]
        
        # Downsample other eval dataset attributes if they exist
        if hasattr(self.eval_dataset, 'metadata'):
            for key, value in self.eval_dataset.metadata.items():
                if isinstance(value, (list, np.ndarray)) and len(value) == original_eval_size:
                    if isinstance(value, list):
                        self.eval_dataset.metadata[key] = [value[i] for i in eval_indices]
                    else:
                        self.eval_dataset.metadata[key] = value[eval_indices]
        
        # Update eval dataset length
        if hasattr(self.eval_dataset, '_dataparser_outputs'):
            # Update image filenames in dataparser outputs if they exist
            if hasattr(self.eval_dataset._dataparser_outputs, 'image_filenames'):
                original_filenames = self.eval_dataset._dataparser_outputs.image_filenames
                if original_filenames is not None:
                    self.eval_dataset._dataparser_outputs.image_filenames = [
                        original_filenames[i] for i in eval_indices
                    ]
        
        # Reset the unseen camera lists with new indices
        self.train_unseen_cameras = list(range(len(train_indices)))
        self.eval_unseen_cameras = list(range(len(eval_indices)))
        
        # Update any depth, normal, and confidence filenames in metadata
        if hasattr(self.train_dataparser_outputs, 'metadata'):
            metadata = self.train_dataparser_outputs.metadata
            
            # List of filename keys that might need updating
            filename_keys = ['depth_filenames', 'sensor_depth_filenames', 
                            'mono_depth_filenames', 'normal_filenames', 'semantic_filenames',
                            'confidence_filenames']
            
            for key in filename_keys:
                if key in metadata and metadata[key] is not None:
                    if len(metadata[key]) == original_train_size:
                        metadata[key] = [metadata[key][i] for i in train_indices]
        
        print(f"Dataset downsampled by factor {scale_factor}:")
        print(f"  Training set: {original_train_size} -> {len(train_indices)} images")
        print(f"  Evaluation set: {original_eval_size} -> {new_eval_size} images")
        if secret_idx is not None and new_secret_idx >= 0:
            print(f"  Secret index preserved: {secret_idx} -> {new_secret_idx}")
        
        return new_secret_idx