import os

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download

from seva.model import Seva, SevaParams

import numpy as np

def seed_everything(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_model(
    model_version: float = 1.1,
    pretrained_model_name_or_path: str = "stabilityai/stable-virtual-camera",
    weight_name: str = "model.safetensors",
    device: str | torch.device = "cuda",
    verbose: bool = False,
) -> Seva:
    if os.path.isdir(pretrained_model_name_or_path):
        weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
    else:
        if model_version > 1:
            base, ext = os.path.splitext(weight_name)
            weight_name = f"{base}v{model_version}{ext}"
        weight_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=weight_name
        )
        _ = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename="config.yaml"
        )

    state_dict = safetensors.torch.load_file(
        weight_path,
        device=str(device),
    )

    with torch.device("meta"):
        model = Seva(SevaParams()).to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if verbose:
        print_load_warning(missing, unexpected)
    return model


# ves poses generation utils
# nerfstudio is opengl convention
def convert_nerfstudio_to_opencv(poses):
    poses = poses.copy()
    poses[:, 2, :] *= -1
    poses = poses[:, np.array([1, 0, 2, 3]), :]
    poses[:, 0:3, 1:3] *= -1
    return poses


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
            # # Skip the (0,0) case because that’s just the original pose.
            # if np.isclose(dtheta, 0.0) and np.isclose(dphi, 0.0):
            #     continue

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


def make_rotation_matrices_opengl(delta_theta, delta_phi):
    """
    Given two angles (in radians), return the 3×3 rotation matrices
    Rx(Δθ) (around X axis) and Ry(Δφ) (around Y axis) for OpenGL convention.
    
    In OpenGL: +X right, +Y up, +Z backward
    
    Rx(Δθ) = [[1,       0,        0     ],
              [0,  cosΔθ,  -sinΔθ ],
              [0,  sinΔθ,   cosΔθ ]]

    Ry(Δφ) = [[ cosΔφ,  0,  sinΔφ ],
              [   0,    1,      0   ],
              [ -sinΔφ,  0,  cosΔφ ]]

    Note: The rotation matrices themselves are the same, but the interpretation
    of the axes differs between OpenGL and OpenCV conventions.

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


def generate_ves_poses_opengl(c2w_secret, angle_limit_degrees=15.0):
    """
    Generate VES poses for OpenGL convention cameras.
    
    In OpenGL convention: +X right, +Y up, +Z backward (camera looks down -Z)
    
    Given a single camera-to-world matrix (4×4) as `c2w_secret` (numpy array),
    produce a list of "VES" camera-to-world matrices by rotating ±angle_limit
    around X and Y axes.

    The key difference from OpenCV is that we need to be careful about the
    sign conventions when rotating around Y axis, since Z points in opposite
    direction.

    Inputs:
    c2w_secret : numpy.ndarray of shape (4, 4)
        - The "secret" camera‐to‐world matrix in OpenGL convention.
    angle_limit_degrees : float
        - The maximum positive/negative rotation (in degrees) to apply around X and Y.

    Returns:
    ves_c2w_list : list of numpy.ndarray of shape (4, 4)
        - A list containing 8 new camera‐to‐world matrices, each rotated by ±δ along X/Y.
    """
    # 1) Convert the secret c2w into w2c (i.e. R_w2c, t_w2c).
    #    If c2w_secret = [ R_c2w | t_c2w ]
    #                    [   0   |    1  ],
    #    then R_w2c = R_c2w^T,  t_w2c = - R_c2w^T @ t_c2w.
    R_c2w = c2w_secret[:3, :3]
    t_c2w = c2w_secret[:3, 3:4]  # shape (3,1)

    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w  # shape (3,1)

    # 2) Build all combinations of Δθ and Δφ in {−δ, 0, +δ}, except (0,0).
    δ_rad = np.deg2rad(angle_limit_degrees)
    deltas = [-δ_rad, 0.0, +δ_rad]

    ves_c2w_list = []

    ves_c2w_list.append(c2w_secret)  # Include the original pose

    for dtheta in deltas:
        for dphi in deltas:
            # Skip the (0,0) case because that's just the original pose.
            if np.isclose(dtheta, 0.0) and np.isclose(dphi, 0.0):
                continue

            # 3) Compute Rx and Ry for these small angles:
            Rx, Ry = make_rotation_matrices_opengl(dtheta, dphi)

            # 4) For OpenGL convention, we apply rotations in the same way:
            #    R'_w2c = Rx @ Ry @ R_w2c
            #    The rotation matrices are mathematically the same, but the
            #    coordinate system interpretation is different.
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
            c2w_prime[:3, 3:4] = tprime_c2w

            ves_c2w_list.append(c2w_prime)

    return ves_c2w_list