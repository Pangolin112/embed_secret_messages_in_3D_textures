import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision


class SobelFilter(nn.Module):
    def __init__(self, ksize=3, use_grayscale=False):
        super(SobelFilter, self).__init__()
        
        # Define Sobel kernels
        if ksize == 3:
            sobel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=torch.float32)
            
            sobel_y = torch.tensor([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]], dtype=torch.float32)
        elif ksize == 5:
            # 5x5 Sobel kernels
            sobel_x = torch.tensor([[-1, -2, 0, 2, 1],
                                    [-4, -8, 0, 8, 4],
                                    [-6, -12, 0, 12, 6],
                                    [-4, -8, 0, 8, 4],
                                    [-1, -2, 0, 2, 1]], dtype=torch.float32)
            
            sobel_y = torch.tensor([[-1, -4, -6, -4, -1],
                                    [-2, -8, -12, -8, -2],
                                    [0, 0, 0, 0, 0],
                                    [2, 8, 12, 8, 2],
                                    [1, 4, 6, 4, 1]], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported kernel size: {ksize}. Use 3 or 5.")
        
        # Reshape for conv2d (out_channels, in_channels, height, width)
        self.sobel_x = sobel_x.view(1, 1, ksize, ksize)
        self.sobel_y = sobel_y.view(1, 1, ksize, ksize)
        
        # Register as buffers (not trainable parameters)
        self.register_buffer('weight_x', self.sobel_x)
        self.register_buffer('weight_y', self.sobel_y)

        self.ksize = ksize
        self.use_grayscale = use_grayscale
        self.padding = ksize // 2
    
    def forward(self, x):
        """
        Apply Sobel filter to input tensor
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Edge magnitude tensor of shape (B, C, H, W)
        """
        if self.use_grayscale and x.shape[1] == 3:
            # Convert to grayscale first
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Ensure weights are on the same device as input
        weight_x = self.weight_x.to(x.device)
        weight_y = self.weight_y.to(x.device)

        # Handle different number of channels
        if x.shape[1] > 1:
            # Apply Sobel to each channel separately
            edges = []
            for i in range(x.shape[1]):
                channel = x[:, i:i+1, :, :]
                gx = F.conv2d(channel, weight_x, padding=self.padding)
                gy = F.conv2d(channel, weight_y, padding=self.padding)
                edge = torch.sqrt(gx**2 + gy**2 + 1e-6)  # Add small epsilon for numerical stability
                edges.append(edge)
            return torch.cat(edges, dim=1)
        else:
            # Single channel
            gx = F.conv2d(x, weight_x, padding=self.padding)
            gy = F.conv2d(x, weight_y, padding=self.padding)
            return torch.sqrt(gx**2 + gy**2 + 1e-6)


class SobelEdgeLoss(nn.Module):
    def __init__(self, loss_type='l1', ksize=3, use_grayscale=False):
        """
        Initialize Sobel Edge Loss
        Args:
            loss_type: 'l1', 'l2', or 'cosine' similarity
        """
        super(SobelEdgeLoss, self).__init__()
        self.sobel = SobelFilter(ksize, use_grayscale)
        self.loss_type = loss_type
        
    def forward(self, pred, target, original_edges, image_dir, step, mask_tensor=None):
        """
        Compute edge-aware loss between predicted and target images
        Args:
            pred: Predicted image tensor (B, C, H, W)
            target: Target image tensor (B, C, H, W)
        Returns:
            Loss value
        """
        # Compute edge maps
        edges_pred = self.sobel(pred)
        edges_target = self.sobel(target)
        if mask_tensor is not None:
            original_edges = original_edges * mask_tensor
            edges_target = edges_target * mask_tensor

        if step % 50 == 0:
            edges_pred_np = edges_pred.detach().cpu()
            edges_target_np = edges_target.detach().cpu()
            original_edges_np = original_edges.detach().cpu()
            added = edges_target_np + original_edges_np
            # Normalize each image individually
            pred_normalized = (edges_pred_np - edges_pred_np.min()) / (edges_pred_np.max() - edges_pred_np.min() + 1e-8)
            target_normalized = (edges_target_np - edges_target_np.min()) / (edges_target_np.max() - edges_target_np.min() + 1e-8)
            original_edges_normalized = (original_edges_np - original_edges_np.min()) / (original_edges_np.max() - original_edges_np.min() + 1e-8)
            added_normalized = (added - added.min()) / (added.max() - added.min() + 1e-8)

            # Save images
            torchvision.utils.save_image(
                pred_normalized, 
                f'{image_dir}/edges_pred_{step}.png'
            )
            torchvision.utils.save_image(
                target_normalized, 
                f'{image_dir}/edges_target.png'
            )
            torchvision.utils.save_image(
                original_edges_normalized, 
                f'{image_dir}/original_edges.png'
            )
            torchvision.utils.save_image(
                added_normalized, 
                f'{image_dir}/added_edges.png'
            )

        # add original edge
        edges_target += original_edges # too shallow target edge
        # edges_target += 0.5 * original_edges
        # mask out the result
        if mask_tensor is not None:
            edges_target = edges_target * mask_tensor
        
        # Compute loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(edges_pred, edges_target)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(edges_pred, edges_target)
        elif self.loss_type == 'cosine':
            # Flatten and compute cosine similarity
            edges_pred_flat = edges_pred.view(edges_pred.shape[0], -1)
            edges_target_flat = edges_target.view(edges_target.shape[0], -1)
            
            # Normalize
            edges_pred_norm = F.normalize(edges_pred_flat, p=2, dim=1)
            edges_target_norm = F.normalize(edges_target_flat, p=2, dim=1)
            
            # Cosine similarity loss (1 - similarity)
            loss = 1.0 - (edges_pred_norm * edges_target_norm).sum(dim=1).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return loss