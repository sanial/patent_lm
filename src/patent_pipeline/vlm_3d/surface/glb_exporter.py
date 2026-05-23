from __future__ import annotations
from pathlib import Path
import trimesh

def export_to_glb(mesh: trimesh.Trimesh, output_dir: Path, file_name: str) -> Path:
    """Export a trimesh object to ``<output_dir>/<file_name>.glb``.

    Args:
        mesh: Mesh to serialize.
        output_dir: Destination directory (created on demand).
        file_name: Base filename (no extension; ``.glb`` is appended).

    Returns:
        Path to the written GLB file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_name}.glb"
    mesh.export(output_path, file_type='glb')
    return output_path
