import torch
try:
    from pytorch3d.utils import ico_sphere
    from pytorch3d.structures import Meshes
except ImportError:
    pass

def make_cylinder(device: str = "cpu") -> "Meshes":
    # Placeholder for actual cylinder generation
    # Since PyTorch3D doesn't have a direct cylinder primitive, we can approximate with a sphere for now,
    # or build vertices/faces manually. For simplicity in the prototype, we'll use ico_sphere.
    return ico_sphere(3, device)

def make_box(device: str = "cpu") -> "Meshes":
    # Placeholder for box
    return ico_sphere(3, device)

def make_sphere(device: str = "cpu") -> "Meshes":
    return ico_sphere(3, device)

def transform_mesh(mesh: "Meshes", dims: list[float], pos: list[float], rot_deg: list[float]) -> "Meshes":
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
