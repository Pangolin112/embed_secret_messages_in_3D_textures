# ves poses generation utils
import numpy as np

# nerfstudio is opengl convention
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