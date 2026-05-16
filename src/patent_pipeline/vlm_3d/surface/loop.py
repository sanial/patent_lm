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
    """
    Surface pattern loop:
    1. Get front view image.
    2. Create a plane mesh textured with the image.
    3. Export to GLB.
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
