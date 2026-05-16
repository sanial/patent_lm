from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .io_utils import ensure_dir


def _run_colmap_if_possible(record: dict, colmap_bin: str, workspace_root: Path) -> tuple[bool, str]:
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
