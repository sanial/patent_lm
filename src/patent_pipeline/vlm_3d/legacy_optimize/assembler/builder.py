import torch
try:
    from pytorch3d.structures import Meshes, join_meshes_as_scene
except ImportError:
    pass

from ...parser.schema import ConstraintSchema
from .primitives import (
    Primitive,
    make_cylinder, make_box, make_sphere, transform_mesh
)

def build_assembly(schema: ConstraintSchema, device: str = "cpu") -> "Meshes":
    meshes_list = []
    
    for part in schema.parts:
        ptype = part.primitive_type.lower()
        if ptype == "cylinder":
            mesh = make_cylinder(device)
        elif ptype == "box":
            mesh = make_box(device)
        else:
            mesh = make_sphere(device)
            
        mesh = transform_mesh(mesh, part.dimensions, part.position, part.rotation_euler_xyz_deg)
        meshes_list.append(mesh)
        
    if not meshes_list:
        return make_sphere(device)
        
    return join_meshes_as_scene(meshes_list)
