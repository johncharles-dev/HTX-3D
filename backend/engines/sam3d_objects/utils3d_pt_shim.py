"""Compatibility shim: utils3d.pt -> utils3d.torch
MoGe expects utils3d.pt; TRELLIS uses utils3d.torch (v0.0.2).
"""
from utils3d.torch import *

import torch

def depth_map_to_point_map(depth, intrinsics):
    """Convert depth map to 3D point map using camera intrinsics."""
    H, W = depth.shape[-2:]
    device = depth.device
    dtype = depth.dtype
    
    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    # Normalize to [0, 1]
    x = (x + 0.5) / W
    y = (y + 0.5) / H
    
    # Intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    
    # Unproject
    z = depth.squeeze(-3) if depth.dim() > 2 else depth
    px = (x - cx[..., None, None]) / fx[..., None, None] * z
    py = (y - cy[..., None, None]) / fy[..., None, None] * z
    
    return torch.stack([px, py, z], dim=-1)
