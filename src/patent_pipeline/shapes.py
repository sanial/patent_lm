from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image


_MODEL_CACHE: dict[str, object] = {}
_INDEX_CACHE: dict[tuple[str, str], tuple[np.ndarray, list[str]]] = {}
_CACHE_LOCK = threading.Lock()


def _primitive_shapes_from_poses(poses: list[dict]) -> list[dict]:
    """Convert pose pseudo-labels into box primitives.

    Used when ``shape.method = primitive``. Each pose's normalized bbox
    dimensions become a ``box`` shape with the same width/height and a
    fixed depth from the pose record.

    Args:
        poses: List of pose dicts produced by the pose stage.

    Returns:
        List of shape dicts (``type='box'``, ``source='primitive_fitting'``).
    """
    shapes: list[dict] = []
    for pose in poses:
        width = float(pose.get("width", 0.1))
        height = float(pose.get("height", 0.1))
        depth = float(pose.get("depth", 0.3))
        shapes.append(
            {
                "instance_id": pose.get("instance_id"),
                "type": "box",
                "dimensions": [width, height, depth],
                "rotation": float(pose.get("rotation", 0.0)),
                "source": "primitive_fitting",
            }
        )
    return shapes


def _clip_retrieval_stub(record: dict, emb_path: str, mesh_index_path: str) -> list[dict]:
    """Deterministic placeholder for CLIP-based ShapeNet retrieval.

    Loads the embedding matrix and mesh index to validate that they exist,
    then assigns mesh index 0 to every instance. Kept around so the JSONL
    schema is stable while real retrieval is being wired up.

    Args:
        record: Manifest row containing ``poses``.
        emb_path: Path to ShapeNet embedding ``.npy`` matrix.
        mesh_index_path: Path to JSON list of ShapeNet mesh ids parallel
            to the embedding rows.

    Returns:
        List of shape dicts (``type='mesh'``, ``source='clip_retrieval_stub'``).

    Raises:
        FileNotFoundError: If either resource file is missing.
    """
    emb_file = Path(emb_path)
    mesh_file = Path(mesh_index_path)
    if not emb_file.exists() or not mesh_file.exists():
        raise FileNotFoundError(
            "clip_retrieval requires valid shapenet_embeddings_npy and shapenet_mesh_index_json paths."
        )

    embeddings = np.load(emb_file)
    with mesh_file.open("r", encoding="utf-8") as f:
        mesh_index = json.load(f)

    poses = record.get("poses", [])
    if not isinstance(poses, list):
        return []

    # Placeholder retrieval: picks index 0 deterministically until image encoder is connected.
    # Keeps schema stable so training code can proceed.
    if len(embeddings) == 0:
        return []
    mesh_id = mesh_index[0] if isinstance(mesh_index, list) and mesh_index else "unknown_mesh"

    out: list[dict] = []
    for pose in poses:
        out.append(
            {
                "instance_id": pose.get("instance_id"),
                "type": "mesh",
                "mesh_id": mesh_id,
                "source": "clip_retrieval_stub",
            }
        )
    return out


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2D matrix (with epsilon guarding).

    Args:
        matrix: 2D array of shape ``[N, D]``.

    Returns:
        Same-shape array whose rows have unit L2 norm. Zero-norm rows are
        clipped to a tiny epsilon to avoid division by zero.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a single vector with epsilon guarding.

    Args:
        vec: 1D array.

    Returns:
        Same-shape unit vector.
    """
    denom = max(float(np.linalg.norm(vec)), 1e-12)
    return vec / denom


def _crop_from_mask(image: np.ndarray, mask: np.ndarray) -> Image.Image | None:
    """Crop the bounding box of a binary mask from an RGB image.

    Args:
        image: RGB image as a ``(H, W, 3)`` ndarray.
        mask: Binary mask as a ``(H, W)`` ndarray; non-zero pixels define
            the region of interest.

    Returns:
        A PIL ``Image`` cropped to the mask's bounding box, or ``None``
        when the mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    crop = image[y_min : y_max + 1, x_min : x_max + 1]
    if crop.size == 0:
        return None
    return Image.fromarray(crop)


def _clip_retrieval_real(
    record: dict,
    emb_path: str,
    mesh_index_path: str,
    clip_model_name: str,
) -> list[dict]:
    """Retrieve nearest ShapeNet meshes by CLIP image similarity.

    Encodes each per-instance crop of the front view with a Sentence-
    Transformers CLIP-style model, then picks the most similar row of the
    pre-built ShapeNet embedding matrix.

    Args:
        record: Manifest row with ``views.front`` and ``masks_path``.
        emb_path: Path to ShapeNet embedding ``.npy`` matrix.
        mesh_index_path: Path to JSON list of mesh ids parallel to the
            embeddings.
        clip_model_name: Sentence-Transformers model identifier.

    Returns:
        List of shape dicts (``type='mesh'``, ``source='clip_retrieval'``)
        with a ``score`` field. Empty list when the front view or masks
        are missing.

    Raises:
        FileNotFoundError: If the embedding or mesh index file is missing.
        RuntimeError: If ``sentence-transformers`` is not installed.
    """
    emb_file = Path(emb_path)
    mesh_file = Path(mesh_index_path)
    if not emb_file.exists() or not mesh_file.exists():
        raise FileNotFoundError(
            "clip_retrieval requires valid shapenet_embeddings_npy and shapenet_mesh_index_json paths."
        )

    shapenet_embeddings, mesh_index = _get_shape_index(emb_file, mesh_file)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers to use shape.method=clip_retrieval") from exc

    views = record.get("views", {})
    if not isinstance(views, dict):
        return []
    front_path = Path(str(views.get("front", "")))
    if not front_path.exists():
        return []

    masks_path = Path(str(record.get("masks_path", "")))
    if not masks_path.exists():
        return []

    front_img = Image.open(front_path).convert("RGB")
    front_arr = np.array(front_img)
    masks = np.load(masks_path)["masks"]

    model = _get_clip_model(clip_model_name)

    out: list[dict] = []
    for idx in range(masks.shape[0]):
        crop = _crop_from_mask(front_arr, masks[idx].astype(np.uint8))
        if crop is None:
            continue

        query_vec = model.encode(crop, convert_to_numpy=True)
        query_vec = _normalize_vec(query_vec)
        similarities = shapenet_embeddings @ query_vec
        best_i = int(np.argmax(similarities))

        out.append(
            {
                "instance_id": idx,
                "type": "mesh",
                "mesh_id": mesh_index[best_i],
                "score": float(similarities[best_i]),
                "source": "clip_retrieval",
            }
        )
    return out


def _get_shape_index(emb_file: Path, mesh_file: Path) -> tuple[np.ndarray, list[str]]:
    """Load (and cache) the ShapeNet embedding matrix and mesh-id list.

    Validates that the JSON index is a non-empty list, the ``.npy`` is 2D,
    and the row count matches. Embeddings are L2-normalized once on load so
    subsequent retrieval is a single matrix multiply.

    Args:
        emb_file: Path to the embedding ``.npy`` file.
        mesh_file: Path to the mesh-id JSON list.

    Returns:
        Tuple ``(embeddings, mesh_index)`` where ``embeddings`` is a
        row-normalized ``(N, D)`` matrix and ``mesh_index`` is the parallel
        list of mesh ids.

    Raises:
        ValueError: If the mesh index is empty or shape mismatches the
            embedding rows.
    """
    cache_key = (str(emb_file.resolve()), str(mesh_file.resolve()))
    with _CACHE_LOCK:
        cached = _INDEX_CACHE.get(cache_key)
        if cached is not None:
            return cached

        with mesh_file.open("r", encoding="utf-8") as f:
            mesh_index = json.load(f)

        if not isinstance(mesh_index, list) or len(mesh_index) == 0:
            raise ValueError("shapenet_mesh_index_json must be a non-empty JSON array.")

        shapenet_embeddings = np.load(emb_file)
        if shapenet_embeddings.ndim != 2:
            raise ValueError("shapenet_embeddings_npy must be a 2D matrix of shape [N, D].")
        if len(mesh_index) != shapenet_embeddings.shape[0]:
            raise ValueError("ShapeNet embedding rows must match mesh index length.")

        shapenet_embeddings = _normalize_rows(shapenet_embeddings)
        _INDEX_CACHE[cache_key] = (shapenet_embeddings, mesh_index)
        return _INDEX_CACHE[cache_key]


def _get_clip_model(clip_model_name: str):
    """Load (and cache) a Sentence-Transformers CLIP-style model by name.

    Args:
        clip_model_name: Hugging Face / Sentence-Transformers identifier.

    Returns:
        A loaded ``SentenceTransformer`` instance.

    Raises:
        RuntimeError: If ``sentence-transformers`` is not installed.
    """
    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(clip_model_name)
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install sentence-transformers to use shape.method=clip_retrieval") from exc
            model = SentenceTransformer(clip_model_name)
            _MODEL_CACHE[clip_model_name] = model
        return model


def prewarm_clip_resources(emb_path: str, mesh_index_path: str, clip_model_name: str) -> None:
    """Eagerly load the ShapeNet index and CLIP model into module caches.

    Useful before launching a multi-record loop so the first record does
    not pay the model-load cost.

    Args:
        emb_path: Path to the ShapeNet embedding ``.npy`` matrix.
        mesh_index_path: Path to the mesh-id JSON list.
        clip_model_name: Sentence-Transformers model identifier.

    Raises:
        FileNotFoundError: If the embedding or mesh index file is missing.
    """
    emb_file = Path(emb_path)
    mesh_file = Path(mesh_index_path)
    if not emb_file.exists() or not mesh_file.exists():
        raise FileNotFoundError(
            "clip_retrieval requires valid shapenet_embeddings_npy and shapenet_mesh_index_json paths."
        )

    _get_shape_index(emb_file, mesh_file)
    _get_clip_model(clip_model_name)


def build_shapes_for_record(
    record: dict,
    method: str,
    emb_path: str,
    mesh_index_path: str,
    clip_model_name: str,
) -> dict:
    """Attach a ``shapes`` field to a manifest row.

    Dispatches to either box-primitive fitting or CLIP-based ShapeNet
    retrieval based on ``method``.

    Args:
        record: Manifest row (must already have ``poses``).
        method: Either ``"primitive"`` or ``"clip_retrieval"``.
        emb_path: ShapeNet embeddings ``.npy`` path (used only for
            ``clip_retrieval``).
        mesh_index_path: ShapeNet mesh-id JSON path.
        clip_model_name: Sentence-Transformers model name.

    Returns:
        A shallow copy of ``record`` extended with a ``shapes`` list.

    Raises:
        ValueError: If ``method`` is not one of the supported values.
    """
    updated = dict(record)
    poses = updated.get("poses", [])
    if not isinstance(poses, list):
        poses = []

    if method == "primitive":
        updated["shapes"] = _primitive_shapes_from_poses(poses)
    elif method == "clip_retrieval":
        updated["shapes"] = _clip_retrieval_real(
            updated,
            emb_path=emb_path,
            mesh_index_path=mesh_index_path,
            clip_model_name=clip_model_name,
        )
    else:
        raise ValueError(f"Unsupported shape.method: {method}")

    return updated
