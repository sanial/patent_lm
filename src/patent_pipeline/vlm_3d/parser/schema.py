from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

class PartSpec(BaseModel):
    part_id: int = Field(description="Unique integer ID for the part (matching the visual label if present)")
    name: str = Field(description="Descriptive name of the part (e.g., 'shaft', 'housing')")
    primitive_type: str = Field(description="One of: 'cylinder', 'box', 'sphere'")
    dimensions: List[float] = Field(description="[width, height, depth] in normalized units (-1.0 to 1.0)")
    position: List[float] = Field(description="[x, y, z] in normalized 3D space relative to center")
    rotation_euler_xyz_deg: List[float] = Field(description="[x, y, z] rotation in degrees")
    material_hint: Optional[str] = Field(default=None, description="Material or texture hint")

class Relation(BaseModel):
    source_part_id: int
    target_part_id: int
    relation_type: str = Field(description="e.g., 'attached_to', 'inside', 'parallel_to'")

class Symmetry(BaseModel):
    part_ids: List[int] = Field(description="List of part IDs that share symmetry")
    symmetry_type: str = Field(description="e.g., 'bilateral', 'radial'")
    reflection_plane: Optional[str] = Field(default=None, description="e.g., 'YZ'")

class ConstraintSchema(BaseModel):
    parts: List[PartSpec]
    relations: List[Relation]
    symmetries: List[Symmetry]
    camera_priors: List[str] = Field(description="e.g., ['front', 'exploded']")
    geometry_hints: List[str] = Field(description="Global hints e.g., ['cylindrical', 'planar']")
