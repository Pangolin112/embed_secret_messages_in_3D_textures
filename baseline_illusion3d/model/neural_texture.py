import torch
import torch.nn as nn
import torch.nn.functional as F


class RGB_field(nn.Module):
    def __init__(self, dtype_half, texture_size, generator):
        super().__init__()

        self.max_resolution = texture_size
        self.dtype = dtype_half
        # Learnable tensor for RGB values (differentiable)
        self.rgb_field_tensor = nn.Parameter(
            torch.zeros(
                self.max_resolution, self.max_resolution, 3, 
                dtype=self.dtype, 
                requires_grad=True
            ), 
            requires_grad=True
        )

        self._initialize_weights()

    def forward(self, uv):
        rgb_field = self.rgb_field_tensor.permute(2, 0, 1).unsqueeze(0)
        uv = uv.view(-1, 1, 1, 2)
        uv_norm = 2.0 * uv - 1.0
        uv_norm_reshaped = uv_norm.permute(1, 0, 2, 3)
        uv_norm_reshaped = uv_norm_reshaped.to(self.rgb_field_tensor.dtype)
        sampled = F.grid_sample(
            input=rgb_field,  # [1,3,H,W]
            grid=uv_norm_reshaped,  # [1,N,1,2]
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        sampled = sampled.squeeze(-1).squeeze(0)  # shape [3, N]
        out = sampled.permute(1, 0)  # shape [N, 3]

        return out

    def _initialize_weights(self):
        with torch.no_grad():
            nn.init.normal_(self.rgb_field_tensor, mean=0.0, std=0.01)
    

class HierarchicalRGB_field(nn.Module):
    def __init__(self, dtype_half, generator, texture_size, num_layers=4):
        super(HierarchicalRGB_field, self).__init__()

        self.layers = nn.ModuleList([RGB_field(dtype_half, texture_size // pow(2, i), generator) for i in range(num_layers)])

    def forward(self, uv):
        y = [layer(uv) for layer in self.layers]
        y = torch.stack(y)
        y = torch.sum(y, dim=0)

        return y
    
    def regularizer(self, weights):
        reg = 0.0

        for i, layer in enumerate(self.layers):
            reg += torch.mean(torch.pow(layer.rgb_field_tensor, 2.0)) * weights[i]

        return reg

    



        