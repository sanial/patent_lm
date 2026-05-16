from __future__ import annotations
import os
import re
from PIL import Image
import json

try:
    from google import genai
except ImportError:
    genai = None


_SELECTOR_MODEL = "gemini-2.5-flash"


def select_best_proxy(
    proxy_image_paths: list[str],
    source_figure_path: str,
    constraint_schema: dict,
) -> str:
    """
    Selects the proxy image that best matches the source patent figure using
    the Gemini API. Returns the path to the best proxy image.
    """
    if genai is None:
        raise ImportError("google-genai package is required to use the proxy selector.")

    if not proxy_image_paths:
        raise ValueError("proxy_image_paths is empty")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    source_img = Image.open(source_figure_path)
    proxy_images = [Image.open(p) for p in proxy_image_paths]

    contents: list = [
        "You are a 3D asset quality control specialist.",
        "Select the photorealistic proxy image that most faithfully represents "
        "the original patent line art in terms of shape, structure, and key "
        "features described in the JSON schema.",
        "JSON Schema:",
        json.dumps(constraint_schema, indent=2),
        "Original line art:",
        source_img,
        "Candidate proxy images follow, in order, starting from index 0:",
    ]
    for i, img in enumerate(proxy_images):
        contents.append(f"Candidate {i}:")
        contents.append(img)
    contents.append(
        "Return ONLY the integer index of the best candidate. No other text."
    )

    try:
        response = client.models.generate_content(
            model=_SELECTOR_MODEL,
            contents=contents,
        )
        text = (response.text or "").strip()
        # Try strict int first, then fall back to first integer found anywhere
        # in the response. Gemini sometimes prepends prose despite instructions.
        try:
            best_index = int(text)
        except ValueError:
            m = re.search(r"\d+", text)
            if not m:
                raise
            best_index = int(m.group(0))
        if 0 <= best_index < len(proxy_image_paths):
            return proxy_image_paths[best_index]
    except (ValueError, IndexError, AttributeError, Exception) as exc:
        print(f"  [proxy_selector] selection failed, defaulting to candidate 0: {exc}")

    return proxy_image_paths[0]

