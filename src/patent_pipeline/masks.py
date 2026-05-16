from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io_utils import ensure_dir


def _extract_instances_from_line_art(image: np.ndarray, min_component_area: int) -> np.ndarray:
    # Patent line drawings are usually dark strokes on bright background.
    _, binary = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    masks: list[np.ndarray] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        instance_mask = (labels == label_id).astype(np.uint8)
        masks.append(instance_mask)

    if not masks:
        return np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
    return np.stack(masks, axis=0)


from .vlm_3d.augmentor.preprocessor import remove_patent_annotations

def build_masks_for_record(record: dict, masks_dir: str | Path, min_component_area: int) -> dict:
    patent_id = str(record.get("patent_id", "unknown"))
    views = record.get("views", {})
    front_path = Path(str(views.get("front", "")))
    if not front_path.exists():
        raise FileNotFoundError(f"Front image not found for patent {patent_id}: {front_path}")

    img = cv2.imread(str(front_path))
    if img is None:
        raise ValueError(f"Failed to read image: {front_path}")

    # Remove annotations before processing
    cleaned_img = remove_patent_annotations(img)
    
    # Convert to grayscale for mask extraction
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)

    instance_masks = _extract_instances_from_line_art(gray, min_component_area=min_component_area)

    out_dir = ensure_dir(masks_dir)
    out_file = out_dir / f"{patent_id}_masks.npz"
    np.savez_compressed(out_file, masks=instance_masks)

    # Save the cleaned image for the augmentation step
    cleaned_image_dir = ensure_dir(Path(masks_dir).parent / "cleaned_images")
    cleaned_image_path = cleaned_image_dir / f"{patent_id}_cleaned.png"
    cv2.imwrite(str(cleaned_image_path), cleaned_img)

    updated = dict(record)
    updated["masks_path"] = str(out_file)
    updated["num_instances"] = int(instance_masks.shape[0])
    updated["cleaned_figure_path"] = str(cleaned_image_path)
    return updated
