from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io_utils import read_jsonl


def _dedupe_preserve(seq: list[str]) -> list[str]:
    """Return ``seq`` with duplicates removed while preserving first-seen order.

    Args:
        seq: Input sequence of strings.

    Returns:
        Deduplicated list with original ordering.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_proxy_mesh_catalog(manifest_rows: list[dict], max_meshes: int) -> list[dict]:
    """Build the proxy mesh catalog used to seed the ShapeNet index.

    Collects unique CPC labels (or titles, when no CPCs are present) from
    the manifest, optionally truncates to ``max_meshes`` entries, and emits
    one record per label with a synthetic ``mesh_id`` and a text caption
    used to compute its CLIP-text embedding.

    Args:
        manifest_rows: Raw manifest rows to mine for labels.
        max_meshes: Cap on entries (0 = no cap).

    Returns:
        List of ``{"mesh_id": ..., "text": ...}`` dicts.
    """
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
    """Compute and persist a CLIP-text proxy shape index from the manifest.

    Encodes the synthetic per-label captions with a Sentence-Transformers
    CLIP-style model and saves the resulting embedding matrix plus a
    parallel mesh-id list. This index is what the ``shape`` stage queries
    when ``method = clip_retrieval``.

    Args:
        manifest_jsonl: Source manifest JSONL.
        output_embeddings_npy: Destination ``.npy`` matrix path.
        output_mesh_index_json: Destination JSON path for the mesh-id list.
        clip_model_name: Sentence-Transformers model identifier.
        max_meshes: Cap on entries (0 = no cap).

    Returns:
        Number of mesh entries written.

    Raises:
        ValueError: If no labels could be extracted from the manifest.
        RuntimeError: If ``sentence-transformers`` is not installed.
    """
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
