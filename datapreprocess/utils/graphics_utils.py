#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
from typing import Generic, Literal, TypeVar, Optional
from jaxtyping import Float
from torch import Tensor, nn


class BasicPointCloud(NamedTuple):
    points : Float[Tensor, "batch dim"]
    colors : Float[Tensor, "batch dim"]
    normals : Optional[Float[Tensor, "batch dim"]] = None
    

def per_face_normals(
    V : torch.Tensor,
    F : torch.Tensor):
    """Compute normals per face.
    
    Args:
        V (torch.FloatTensor): Vertices of shape [V, 3]
        F (torch.LongTensor): Faces of shape [F, 3]
    
    Returns:
        (torch.FloatTensor): Normals of shape [F, 3]
    """
    mesh = V[F]

    vec_a = mesh[:, 0] - mesh[:, 1]
    vec_b = mesh[:, 2] - mesh[:, 1]
    normals = torch.cross(vec_a, vec_b, dim=-1)
    return torch.nn.functional.normalize(normals, eps=1e-6, dim=1)


def per_vertex_normals(
    V : torch.Tensor,
    F : torch.Tensor):
    """Compute normals per face.
    
    Args:
        V (torch.FloatTensor): Vertices of shape [V, 3]
        F (torch.LongTensor): Faces of shape [F, 3]
    
    Returns:
        (torch.FloatTensor): Normals of shape [F, 3]
    """
    verts_normals = torch.zeros_like(V)
    mesh = V[F]

    faces_normals = torch.cross(
        mesh[:, 2] - mesh[:, 1],
        mesh[:, 0] - mesh[:, 1],
        dim=1,
    )

    verts_normals.index_add_(0, F[:, 0], faces_normals)
    verts_normals.index_add_(0, F[:, 1], faces_normals)
    verts_normals.index_add_(0, F[:, 2], faces_normals)
    
    return torch.nn.functional.normalize(
        verts_normals, eps=1e-6, dim=1
    )
    
def get_vertex_mass(vertices, faces, density=0.2):
    '''
    Computes the mass of each vertex according to triangle areas and fabric density
    '''
    # Get areas of triangles
    areas = get_face_areas(vertices, faces)
    triangle_masses = density * areas
    
    # Initialize vertex masses
    vertex_masses = torch.zeros(vertices.shape[0], device=vertices.device)
    
    # Distribute triangle masses to vertices
    # Similar to np.add.at but using scatter_add_
    vertex_masses.scatter_add_(0, faces[:, 0], triangle_masses / 3)
    vertex_masses.scatter_add_(0, faces[:, 1], triangle_masses / 3)
    vertex_masses.scatter_add_(0, faces[:, 2], triangle_masses / 3)
    
    return vertex_masses

def get_face_areas(vertices, faces):
    '''
    Compute areas of triangular faces
    '''
    # Get vertices of triangles
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    
    # Compute edge vectors
    e1 = v2 - v1
    e2 = v3 - v1
    
    # Compute cross product and get area
    cross = torch.cross(e1, e2, dim=1)
    areas = 0.5 * torch.norm(cross, dim=1)
    
    return areas


def rotate_vector_to_vector(v1, v2):
    """
    Returns a rotation matrix that rotates v1 to align with v2.
    """
    assert v1.dim() == v2.dim()
    assert v1.shape[-1] == 3
    if v1.dim() == 1:
        v1 = v1[None, ...]
        v2 = v2[None, ...]
    N = v1.shape[0]

    u = v1 / torch.norm(v1, dim=-1, keepdim=True)
    Ru = v2 / torch.norm(v2, dim=-1, keepdim=True)
    I = torch.eye(3, 3, device=v1.device).unsqueeze(0).repeat(N, 1, 1)

    # the cos angle between the vectors
    c = torch.bmm(u.view(N, 1, 3), Ru.view(N, 3, 1)).squeeze(-1)

    eps = 1.0e-10
    # the cross product matrix of a vector to rotate around
    K = torch.bmm(Ru.unsqueeze(2), u.unsqueeze(1)) - torch.bmm(
        u.unsqueeze(2), Ru.unsqueeze(1)
    )
    # Rodrigues' formula
    ans = I + K + (K @ K) / (1 + c)[..., None]
    same_direction_mask = torch.abs(c - 1.0) < eps
    same_direction_mask = same_direction_mask.squeeze(-1)
    opposite_direction_mask = torch.abs(c + 1.0) < eps
    opposite_direction_mask = opposite_direction_mask.squeeze(-1)
    ans[same_direction_mask] = torch.eye(3, device=v1.device)
    ans[opposite_direction_mask] = -torch.eye(3, device=v1.device)
    return ans

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View2_torch(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = torch.tensor(translate, dtype=torch.float32)
    
    # Initialize the transformation matrix
    Rt = torch.zeros((4, 4), dtype=torch.float32)
    Rt[:3, :3] = R.t()  # Transpose of R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # Compute the inverse to get the camera-to-world transformation
    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    
    # Invert again to get the world-to-view transformation
    Rt = torch.linalg.inv(C2W)
    
    return Rt

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))