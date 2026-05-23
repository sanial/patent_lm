from __future__ import annotations
from pathlib import Path
from .plane_builder import create_textured_plane
from .glb_exporter import export_to_glb

def run_surface_pattern_loop(
    record: dict,
    config: "Vlm3dConfig",
    work_dir: Path,
    device: str = "cuda",
) -> dict:
    """Reconstruct a surface-pattern patent as a textured plane.

    For CPC classes that describe 2D patterns (textile, surface ornament),
    full 3D reconstruction is meaningless. Instead, the front view image is
    pasted as a texture onto a thin box and exported to GLB.

    Args:
        record: Manifest row (must include ``views.front`` and
            ``patent_id``).
        config: VLM-3D configuration block (unused here but kept for
            interface symmetry with the art3d loop).
        work_dir: Directory whose ``surface_meshes/`` sub-folder will
            receive the GLB.
        device: Compute device label (unused; surface meshing is CPU-only).

    Returns:
        A shallow copy of ``record`` with ``surface_result`` and (on
        success) ``mesh_path``.
    """
    print(f"Running surface pattern loop for {record['patent_id']}")
    updated = dict(record)
    
    front_view_path = record.get("views", {}).get("front")
    if not front_view_path or not Path(front_view_path).exists():
        updated["surface_result"] = "failed: no front view image"
        return updated

    # 1. Create textured plane
    plane_mesh = create_textured_plane(front_view_path)
    
    # 2. Export to GLB
    output_dir = work_dir / "surface_meshes"
    glb_path = export_to_glb(plane_mesh, output_dir, record['patent_id'])
    
    updated["surface_result"] = "processed"
    updated["mesh_path"] = str(glb_path)
    
    return updated
