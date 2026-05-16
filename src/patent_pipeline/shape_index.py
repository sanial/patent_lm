from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io_utils import read_jsonl


def _dedupe_preserve(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_proxy_mesh_catalog(manifest_rows: list[dict], max_meshes: int) -> list[dict]:
    labels: list[str] = []
    for row in manifest_rows:
        cpc = row.get("cpc")
        if isinstance(cpc, list):
            for code in cpc:
                if isinstance(code, str) and code.strip():
                    labels.append(code.strip())

    labels = _dedupe_preserve(labels)
    if not labels:
        # Fallback to titles if class labels are absent.
        for row in manifest_rows:
            title = row.get("title")
            if isinstance(title, str) and title.strip():
                labels.append(title.strip())
        labels = _dedupe_preserve(labels)

    if max_meshes > 0:
        labels = labels[:max_meshes]

    catalog: list[dict] = []
    for label in labels:
        mesh_id = f"proxy::{label.replace(' ', '_')}"
        text = f"3D mesh of product design class {label}"
        catalog.append({"mesh_id": mesh_id, "text": text})
    return catalog


def build_proxy_shape_index_from_manifest(
    manifest_jsonl: str | Path,
    output_embeddings_npy: str | Path,
    output_mesh_index_json: str | Path,
    clip_model_name: str,
    max_meshes: int = 0,
) -> int:
    rows = read_jsonl(manifest_jsonl)
    catalog = _extract_proxy_mesh_catalog(rows, max_meshes=max_meshes)
    if not catalog:
        raise ValueError("No labels found in manifest to build proxy shape index.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers to build proxy shape index") from exc

    model = SentenceTransformer(clip_model_name)
    texts = [item["text"] for item in catalog]
    embeddings = model.encode(texts, convert_to_numpy=True)

    emb_path = Path(output_embeddings_npy)
    mesh_path = Path(output_mesh_index_json)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(emb_path, embeddings)
    mesh_index = [item["mesh_id"] for item in catalog]
    mesh_path.write_text(json.dumps(mesh_index, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(mesh_index)
