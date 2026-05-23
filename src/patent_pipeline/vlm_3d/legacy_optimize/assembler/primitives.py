import torch
try:
    from pytorch3d.utils import ico_sphere
    from pytorch3d.structures import Meshes
except ImportError:
    pass

def make_cylinder(device: str = "cpu") -> "Meshes":
    """Return a placeholder cylinder primitive (currently an icosphere).

    PyTorch3D ships no native cylinder generator, so the prototype uses
    an :func:`ico_sphere` at subdivision level 3 as a stand-in.

    Args:
        device: Torch device string.

    Returns:
        A :class:`pytorch3d.structures.Meshes`.
    """
    # Placeholder for actual cylinder generation
    # Since PyTorch3D doesn't have a direct cylinder primitive, we can approximate with a sphere for now,
    # or build vertices/faces manually. For simplicity in the prototype, we'll use ico_sphere.
    return ico_sphere(3, device)

def make_box(device: str = "cpu") -> "Meshes":
    """Return a placeholder box primitive (currently an icosphere).

    Args:
        device: Torch device string.

    Returns:
        A :class:`pytorch3d.structures.Meshes`.
    """
    # Placeholder for box
    return ico_sphere(3, device)

def make_sphere(device: str = "cpu") -> "Meshes":
    """Return an icosphere primitive at subdivision level 3.

    Args:
        device: Torch device string.

    Returns:
        A :class:`pytorch3d.structures.Meshes`.
    """
    return ico_sphere(3, device)

def transform_mesh(mesh: "Meshes", dims: list[float], pos: list[float], rot_deg: list[float]) -> "Meshes":
    """Apply per-axis scale and translation to a primitive mesh.

    Rotation is currently ignored in the prototype skeleton.

    Args:
        mesh: Source mesh with a single batch element.
        dims: Per-axis scale factors ``[sx, sy, sz]``.
        pos: Translation ``[tx, ty, tz]``.
        rot_deg: Euler rotation ``[rx, ry, rz]`` in degrees (currently
            unused; reserved for future use).

    Returns:
        A new :class:`pytorch3d.structures.Meshes` with transformed
        vertices.
    """
    # Apply scale, then rotation, then translation
    # Scale
    verts = mesh.verts_list()[0]
    scale = torch.tensor(dims, dtype=torch.float32, device=verts.device)
    verts = verts * scale
    
    # Rotation (simplified: we'll ignore complex euler for the bare skeleton)
    # Translation
    trans = torch.tensor(pos, dtype=torch.float32, device=verts.device)
    verts = verts + trans
    
    return Meshes(verts=[verts], faces=mesh.faces_list())
