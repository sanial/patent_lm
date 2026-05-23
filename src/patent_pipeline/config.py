from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    raw_manifest: str
    work_dir: str
    output_dir: str


@dataclass
class FilterConfig:
    target_cpc_prefixes: list[str]
    max_samples: int


@dataclass
class MasksConfig:
    min_component_area: int
    workers: int = 1


@dataclass
class PoseConfig:
    use_colmap: bool
    colmap_bin: str
    fixed_depth: float
    workers: int = 1


@dataclass
class ShapeConfig:
    method: str
    shapenet_embeddings_npy: str
    shapenet_mesh_index_json: str
    clip_model_name: str = "clip-ViT-B-32"
    workers: int = 1


@dataclass
class RuntimeConfig:
    max_retries: int = 1


@dataclass
class PackagingConfig:
    build_hf_dataset: bool
    hf_dataset_dir: str
    output_jsonl: str


@dataclass
class Vlm3dConfig:
    vllm_endpoint: str
    model_name: str
    optimization_steps: int
    num_candidates: int
    loss_weights: dict

@dataclass
class PipelineConfig:
    paths: PathsConfig
    filter: FilterConfig
    masks: MasksConfig
    pose: PoseConfig
    shape: ShapeConfig
    packaging: PackagingConfig
    runtime: RuntimeConfig
    vlm_3d: Vlm3dConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file into a dict, validating the top-level type.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Parsed mapping.

    Raises:
        ValueError: If the parsed document is not a mapping.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not parse to a dict.")
    return data


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline YAML configuration.

    Args:
        path: Path to the YAML file (typically ``configs/pipeline.yaml``).

    Returns:
        A fully-typed :class:`PipelineConfig` populated from the YAML.
    """
    cfg_path = Path(path)
    raw = _load_yaml(cfg_path)

    return PipelineConfig(
        paths=PathsConfig(**raw["paths"]),
        filter=FilterConfig(**raw["filter"]),
        masks=MasksConfig(**raw["masks"]),
        pose=PoseConfig(**raw["pose"]),
        shape=ShapeConfig(**raw["shape"]),
        packaging=PackagingConfig(**raw["packaging"]),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        vlm_3d=Vlm3dConfig(**raw.get("vlm_3d", {})),
    )
