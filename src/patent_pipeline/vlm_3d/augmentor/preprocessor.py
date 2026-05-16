from __future__ import annotations

import easyocr
import cv2
import numpy as np

def remove_patent_annotations(img_array: np.ndarray) -> np.ndarray:
    """Mask out text/annotations before edge extraction."""
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(img_array)
    mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    for (bbox, text, conf) in results:
        if conf > 0.4:
            pts = np.array(bbox, dtype=np.int32)
            # dilate slightly to catch leader lines near text
            cv2.fillPoly(mask, [pts], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel)
    inpainted = cv2.inpaint(img_array, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted
