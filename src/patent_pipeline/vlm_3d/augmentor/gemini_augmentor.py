from __future__ import annotations
import os
from pathlib import Path
from PIL import Image
import io

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


_IMAGE_GEN_MODEL = "gemini-2.5-flash-image"


def augment_figure_with_gemini(
    cleaned_figure_path: str,
    diffusion_prompt: str,
    n_candidates: int = 4,
) -> list[Image.Image]:
    """
    Generates N proxy images from a cleaned patent figure using the Gemini
    image-generation API. Acts as a stand-in for a local ControlNet pipeline.
    """
    if genai is None:
        raise ImportError("google-genai package is required to use the Gemini augmentor.")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)
    source_img = Image.open(cleaned_figure_path)

    instruction = (
        "You are a photorealistic rendering engine. Given a patent line-art "
        "figure and a text description, produce a photorealistic product photo "
        "of the same object, preserving its geometry, proportions, and "
        "viewpoint. Plain neutral background, soft studio lighting.\n\n"
        f"Description: {diffusion_prompt}"
    )

    contents = [instruction, source_img]

    images: list[Image.Image] = []
    for i in range(n_candidates):
        try:
            response = client.models.generate_content(
                model=_IMAGE_GEN_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )
        except Exception as exc:
            print(f"  [augmentor] candidate {i} failed: {exc}")
            continue

        candidate = (response.candidates or [None])[0]
        if candidate is None or candidate.content is None:
            continue

        for part in candidate.content.parts or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and inline.data:
                try:
                    pil_img = Image.open(io.BytesIO(inline.data))
                    images.append(pil_img)
                    break
                except Exception as exc:
                    print(f"  [augmentor] failed to decode image: {exc}")

    return images

