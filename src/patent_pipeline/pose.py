from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .io_utils import ensure_dir


def _run_colmap_if_possible(record: dict, colmap_bin: str, workspace_root: Path) -> tuple[bool, str]:
    """Best-effort attempt to run COLMAP automatic reconstruction.

    Patent figures rarely have enough overlapping views for structure-from-
    motion, so this helper degrades gracefully when COLMAP cannot run or
    exits non-zero. Used as a feasibility probe rather than a hard
    dependency.

    Args:
        record: Manifest row (uses the ``views`` dict and ``patent_id``).
        colmap_bin: Path or command name of the COLMAP executable.
        workspace_root: Parent directory under which per-patent COLMAP
            workspaces are created.

    Returns:
        A ``(ok, info)`` tuple. ``ok`` is True only when COLMAP exited
        with code 0. ``info`` is the workspace path on success or a
        truncated error message / status string on failure.
    """
    views = record.get("views", {})
    if not isinstance(views, dict):
        return False, "No views dict."

    image_paths = [Path(str(p)) for p in views.values() if isinstance(p, str) and p.strip()]
    image_paths = [p for p in image_paths if p.exists()]
    if len(image_paths) < 3:
        return False, "Not enough valid views for COLMAP."

    patent_id = str(record.get("patent_id", "unknown"))
    patent_dir = ensure_dir(workspace_root / patent_id)

    cmd = [
        colmap_bin,
        "automatic_reconstructor",
        "--image_path",
        str(image_paths[0].parent),
        "--workspace_path",
        str(patent_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()[:500]
    return True, str(patent_dir)


def _poses_from_masks(front_image_path: Path, masks_path: Path, fixed_depth: float) -> list[dict]:
    """Derive 2.5D pose pseudo-labels from instance masks on the front view.

    For each mask, computes the axis-aligned bounding box and emits a pose
    record with normalized ``(x, y)`` center, normalized ``width`` and
    ``height``, a placeholder ``z = -1.0``, and a configurable
    ``depth``/``rotation``.

    Args:
        front_image_path: Path to the patent's front view image.
        masks_path: Path to a ``.npz`` archive whose ``masks`` array has
            shape ``(N, H, W)``.
        fixed_depth: Constant depth value to assign to every pose since
            real depth is unknown from a single line drawing.

    Returns:
        List of pose dicts with normalized geometry, ready to be attached
        to the manifest row.

    Raises:
        ValueError: If the front image cannot be read.
    """
    image = cv2.imread(str(front_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {front_image_path}")

    h, w = image.shape[:2]
    data = np.load(masks_path)
    masks = data["masks"]

    poses: list[dict] = []
    for idx in range(masks.shape[0]):
        mask = masks[idx].astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        bbox_w = max(1, x_max - x_min)
        bbox_h = max(1, y_max - y_min)
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        poses.append(
            {
                "instance_id": idx,
                "x": center_x / w,
                "y": center_y / h,
                "z": -1.0,
                "width": bbox_w / w,
                "height": bbox_h / h,
                "depth": fixed_depth,
                "rotation": 0.0,
            }
        )

    return poses


def build_poses_for_record(
    record: dict,
    use_colmap: bool,
    colmap_bin: str,
    colmap_workspace_root: str | Path,
    fixed_depth: float,
) -> dict:
    """Compute pose pseudo-labels for one manifest row.

    Optionally probes COLMAP (recording the outcome) and then always
    derives mask-based 2.5D poses for downstream stages.

    Args:
        record: Manifest row with ``views`` and ``masks_path`` populated.
        use_colmap: If True, attempt COLMAP first and record its status.
        colmap_bin: COLMAP executable path/name.
        colmap_workspace_root: Directory to host per-patent COLMAP
            workspaces.
        fixed_depth: Constant depth passed to :func:`_poses_from_masks`.

    Returns:
        A shallow copy of ``record`` augmented with ``poses`` (always) and,
        when ``use_colmap`` is True, ``colmap_ok``/``colmap_info``.

    Raises:
        FileNotFoundError: If the front view or masks archive is missing.
    """
    updated = dict(record)

    if use_colmap:
        ok, info = _run_colmap_if_possible(record, colmap_bin, Path(colmap_workspace_root))
        updated["colmap_ok"] = ok
        updated["colmap_info"] = info

    views = updated.get("views", {})
    front_path = Path(str(views.get("front", "")))
    masks_path = Path(str(updated.get("masks_path", "")))
    if not front_path.exists() or not masks_path.exists():
        raise FileNotFoundError("Both front image and masks_path must exist before pose stage.")

    updated["poses"] = _poses_from_masks(front_path, masks_path, fixed_depth=fixed_depth)
    return updated
