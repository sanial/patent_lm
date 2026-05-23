from __future__ import annotations
import trimesh
import numpy as np

def create_textured_plane(image_path: str) -> trimesh.Trimesh:
    """Build a thin trimesh box and apply ``image_path`` as its base color.

    Used by the surface-pattern loop to represent 2D textile/ornament
    patents as a textured planar mesh.

    Args:
        image_path: Path to the image used as the PBR base-color texture.

    Returns:
        A :class:`trimesh.Trimesh` (a thin box) with UV coordinates and
        either a PBR or color visual attached.
    """
    # Create a simple plane mesh
    plane = trimesh.creation.box(extents=[1, 1, 0.01])
    
    # Create material
    try:
        from PIL import Image
        material = trimesh.visual.texture.PBRMaterial(
            baseColorTexture=Image.open(image_path)
        )
    except ImportError:
        # Fallback if Pillow is not installed (should be, via other deps)
        material = trimesh.visual.ColorVisuals()

    # Apply the material to the mesh
    plane.visual = trimesh.visual.TextureVisuals(
        uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        material=material
    )
    
    # We need to manually assign the UV coordinates to the correct vertices.
    # This is a simplified mapping for a simple box.
    # A real implementation would need more robust UV mapping.
    face_uv = np.array([
        [0, 1], [1, 1], [1, 0], # one face
        [0, 1], [1, 0], [0, 0]  # other face
    ])
    
    # Find the two large faces of the box
    large_faces = [i for i, normal in enumerate(plane.face_normals) if np.allclose(np.abs(normal), [0, 0, 1])]
    
    if len(large_faces) >= 2:
        # A bit of a hack: assign UVs to the two main faces
        # This assumes a standard box creation order.
        visual = trimesh.visual.TextureVisuals(
            uv=np.array([[0,0], [1,0], [1,1], [0,1], [0,0], [1,0], [1,1], [0,1]]),
            material=material,
        )
        plane.visual = visual

    return plane
