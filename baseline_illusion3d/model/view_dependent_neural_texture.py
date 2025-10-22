import torch
import torch.nn as nn
import torch.nn.functional as F


# SH coefficients 
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]

# ref https://github.com/sxyu/svox2/blob/ee80e2c4df8f29a407fda5729a494be94ccf9234/svox2/utils.py#L115C36-L115C40
def eval_sh_bases(basis_dim : int, dirs : torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y
        result[..., 2] = SH_C1 * z
        result[..., 3] = -SH_C1 * x
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy
            result[..., 5] = SH_C2[1] * yz
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = SH_C2[3] * xz
            result[..., 8] = SH_C2[4] * (xx - yy)

    return result


def convert_to_ndc(origins, directions, ndc_coeffs, near: float = 1.0):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane, not sure if needed
    t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = directions.unbind(-1)
    ox, oy, oz = origins.unbind(-1)

    # Projection
    o0 = ndc_coeffs[0] * (ox / oz)
    o1 = ndc_coeffs[1] * (oy / oz)
    o2 = 1 - 2 * near / oz

    d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
    d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
    d2 = 2 * near / oz;

    origins = torch.stack([o0, o1, o2], -1)
    directions = torch.stack([d0, d1, d2], -1)

    return origins, directions


def gen_rays(c2w, height, width, cx_val, cy_val, fx_val, fy_val, ndc_coeffs):
    """
    Generate the rays for this camera
    :return: (origins (H*W, 3), dirs (H*W, 3))
    """
    origins = c2w[None, :3, 3].expand(height * width, -1).contiguous()
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=c2w.device) + 0.5,
        torch.arange(width, dtype=torch.float64, device=c2w.device) + 0.5,
    )
    xx = (xx - cx_val) / fx_val
    yy = (yy - cy_val) / fy_val
    zz = torch.ones_like(xx)
    dirs = torch.stack((xx, yy, zz), dim=-1)   # OpenCV
    del xx, yy, zz
    dirs /= torch.norm(dirs, dim=-1, keepdim=True)
    dirs = dirs.reshape(-1, 3, 1)
    dirs = (c2w[None, :3, :3].double() @ dirs)[..., 0]
    dirs = dirs.reshape(-1, 3).float()

    if ndc_coeffs[0] > 0.0:
        origins, dirs = convert_to_ndc(
                origins,
                dirs,
                ndc_coeffs)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)

    return origins, dirs


def eval_sh_basis(viewdirs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate 9 spherical harmonic (SH) basis functions for given view directions.
    Assumes viewdirs has shape (..., 3) with components (x, y, z) already normalized.

    Returns:
        A tensor of shape (..., 9) with SH coefficients.
    """
    x = viewdirs[..., 0]
    y = viewdirs[..., 1]
    z = viewdirs[..., 2]
    
    # # Degree 0.
    # sh0 = 0.282095 * torch.ones_like(x)  # Constant term
    
    # # Degree 1.
    # sh1 = 0.488603 * y  # Y_1^{-1}
    # sh2 = 0.488603 * z  # Y_1^{0}
    # sh3 = 0.488603 * x  # Y_1^{1}
    
    # # Degree 2.
    # sh4 = 1.092548 * x * y              # Y_2^{-2}
    # sh5 = 1.092548 * y * z              # Y_2^{-1}
    # sh6 = 0.315392 * (3 * z * z - 1)      # Y_2^{0}
    # sh7 = 1.092548 * x * z              # Y_2^{1}
    # sh8 = 0.546274 * (x * x - y * y)      # Y_2^{2}

    # Degree 0.
    sh0 = SH_C0 * torch.ones_like(x)  # Constant term
    # Degree 1.
    sh1 = -SH_C1 * y  
    sh2 = SH_C1 * z
    sh3 = -SH_C1 * x
    # Degree 2.
    sh4 = SH_C2[0] * x * y
    sh5 = SH_C2[1] * y * z
    sh6 = SH_C2[2] * (2.0 * z * z - x * x - y * y)
    sh7 = SH_C2[3] * x * z
    sh8 = SH_C2[4] * (x * x - y * y)
    
    # Stack into the last dimension (order: [sh0, sh1, ..., sh8]).
    sh_basis = torch.stack([sh0, sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8], dim=-1)
    return sh_basis


class ViewDependentTexture(nn.Module):
    def __init__(self, height: int, width: int, basis_dim: int = 9):
        """
        Initialize the view-dependent texture module.
        
        Args:
            height: Texture height (number of rows).
            width: Texture width (number of columns).
            basis_dim: Dimensionality of the SH basis (default is 9).
        """
        super(ViewDependentTexture, self).__init__()
        self.height = height
        self.width = width
        self.basis_dim = basis_dim
        
        # Create a learnable texture parameter of shape (H, W, 3, basis_dim).
        # It represents per-pixel SH coefficients for 3 channels.
        # Here we initialize with a small random value.
        texture = torch.randn((height, width, 3, basis_dim)) * 0.01
        
        # Reshape to be compatible with F.grid_sample:
        # Rearranging to shape (1, 3 * basis_dim, H, W).
        texture = texture.view(height, width, 3 * basis_dim).permute(2, 0, 1).unsqueeze(0)
        self.texture = nn.Parameter(texture)

    def forward(self, uv: torch.Tensor, viewdirs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the view-dependent texture module.
        
        Args:
            uv: Tensor of UV coordinates for texture sampling.
                Expected shape is (B, H, W, 2), normalized to [-1, 1].
            viewdirs: Tensor of view directions in world space.
                Expected shape is (B, H, W, 3) (assumed normalized).
        
        Returns:
            rgb: View-dependent RGB image as a tensor of shape (B, 3, H, W).
        """
        B, H, W, _ = uv.shape

        # Compute the SH basis for each view-direction.
        # The resulting tensor has shape (H, W, 9).
        sh_basis = eval_sh_basis(viewdirs)
        sh_basis = sh_basis.unsqueeze(0)
        # Expand sh_basis from (B, H, W, 9) to (B, H, W, 1, 9) for broadcasting.
        sh_basis_exp = sh_basis.unsqueeze(3)
        
        # Sample the texture using the provided uv grid.
        # grid_sample expects the grid with shape (B, H, W, 2) and the texture in shape (N, C, H, W).
        # Our texture is stored in self.texture with shape (1, 3*9, H, W).
        # If B > 1, the texture is shared among the batch elements.
        sh_coeff = F.grid_sample(self.texture, uv, align_corners=True)  # shape (1, 3*9, H, W)
        if sh_coeff.shape[0] == 1 and B > 1:
            sh_coeff = sh_coeff.expand(B, -1, -1, -1)
        
        # Correctly reshape the sampled texture.
        # Current sh_coeff shape is (B, 3*9, H, W) where the ordering is assumed to be:
        # [Red's 9 coefficients, Green's 9 coefficients, Blue's 9 coefficients].
        # We want to reshape such that for each pixel we get (3, 9) with 9 coefficients per channel.
        sh_coeff = sh_coeff.view(B, self.basis_dim, 3, H, W)   # results in shape (B, 9, 3, H, W)
        sh_coeff = sh_coeff.permute(0, 3, 4, 2, 1)              # now shape is (B, H, W, 3, 9)
        
        # Elementwise multiplication followed by summing over the basis dimension.
        rgb = (sh_coeff * sh_basis_exp).sum(dim=-1)  # resulting shape: (B, H, W, 3)

        return rgb

    
class HierarchicalSH_field(nn.Module):
    def __init__(self, texture_size, basis_dim, num_layers=4):
        super(HierarchicalSH_field, self).__init__()

        self.layers = nn.ModuleList([ViewDependentTexture(texture_size // pow(2, i), texture_size // pow(2, i), basis_dim) for i in range(num_layers)])

    def forward(self, uv, rays):
        y = [layer(uv, rays) for layer in self.layers]
        y = torch.stack(y)
        y = torch.sum(y, dim=0)

        return y
    
    def regularizer(self, weights):
        reg = 0.0

        for i, layer in enumerate(self.layers):
            reg += torch.mean(torch.pow(layer.texture, 2.0)) * weights[i]

        return reg