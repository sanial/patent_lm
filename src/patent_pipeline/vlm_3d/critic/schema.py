from typing import List
from pydantic import BaseModel, Field

class Issue(BaseModel):
    part: str
    type: str
    views: List[str]

class CriticSchema(BaseModel):
    view_consistency: float = Field(ge=0.0, le=1.0)
    symmetry_ok: bool
    boundary_fidelity: float = Field(ge=0.0, le=1.0)
    issues: List[Issue]
    rerank_score: float = Field(ge=0.0, le=1.0)
    recommended_actions: List[str]
