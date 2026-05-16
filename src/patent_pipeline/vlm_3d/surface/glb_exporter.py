from __future__ import annotations
from pathlib import Path
import trimesh

def export_to_glb(mesh: trimesh.Trimesh, output_dir: Path, file_name: str) -> Path:
    """
    Exports a trimesh object to a GLB file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_name}.glb"
    mesh.export(output_path, file_type='glb')
    return output_path
