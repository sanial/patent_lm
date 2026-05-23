from __future__ import annotations

import easyocr
import cv2
import numpy as np

def remove_patent_annotations(img_array: np.ndarray) -> np.ndarray:
    """Mask out text/annotations from a patent figure before edge extraction.

    Runs EasyOCR on the image, fills the detected text regions (with a
    morphological dilation to catch nearby leader lines), and inpaints over
    them via OpenCV's ``INPAINT_TELEA`` algorithm.

    Args:
        img_array: Input image as a BGR (or grayscale) ``ndarray``.

    Returns:
        Same-shape image with reference numbers and labels erased.
    """
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
