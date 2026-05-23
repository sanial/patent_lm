import torch
try:
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, look_at_view_transform, 
        RasterizationSettings, MeshRasterizer, MeshRenderer,
        SoftSilhouetteShader, BlendParams
    )
except ImportError:
    pass
import numpy as np


def get_camera_rig(device: str = "cpu"):
    """Build a four-view PyTorch3D camera rig (front, side, top, iso).

    Args:
        device: Torch device string.

    Returns:
        A :class:`FoVPerspectiveCameras` instance with batch size 4.
    """
    # Create 4 views: front, side, top, iso
    dist = 3.0
    elev = torch.tensor([0.0, 0.0, 90.0, 30.0])
    azim = torch.tensor([0.0, 90.0, 0.0, 45.0])
    
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    return cameras

def get_silhouette_renderer(device: str = "cpu", image_size: int = 256):
    """Build a PyTorch3D soft-silhouette renderer paired with the 4-view rig.

    Args:
        device: Torch device string.
        image_size: Square render resolution.

    Returns:
        A :class:`pytorch3d.renderer.MeshRenderer` configured for
        soft-silhouette rasterization.
    """
    cameras = get_camera_rig(device)
    
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=np.log(1. / 1e-4 - 1.) * 1e-4, # soft edges
        faces_per_pixel=50, 
    )
    
    silhouette_shader = SoftSilhouetteShader(
        blend_params=BlendParams(sigma=1e-4, gamma=1e-4)
    )
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=silhouette_shader
    )
    
    return renderer
