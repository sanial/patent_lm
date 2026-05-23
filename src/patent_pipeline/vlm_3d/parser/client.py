import os
from pathlib import Path
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

from .schema import ConstraintSchema

def parse_constraints(
    image_paths: list[Path],
    caption: str,
    model_name: str = "gemini-1.5-pro",
) -> ConstraintSchema:
    """Extract a 3D constraint schema from patent figures via a VLM.

    Sends the provided figures and caption to the named Gemini model with
    a JSON-schema response constraint so the reply parses straight into a
    :class:`ConstraintSchema`.

    Args:
        image_paths: List of patent figure paths; missing files are
            silently skipped.
        caption: Patent caption or title to give the model textual context.
        model_name: Gemini model identifier (default ``gemini-1.5-pro``).

    Returns:
        A populated :class:`ConstraintSchema`.

    Raises:
        ValueError: If the response could not be coerced into the schema.
    """
    
    api_key = os.environ.get("GEMINI_API_KEY", "")
    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are a 3D modeling expert analyzing patent diagrams.
    Given the following patent figures and the caption: '{caption}',
    extract the 3D structure and constraints.
    Provide the exact 3D layout, parts, symmetries, and relations.
    """
    
    images = []
    for path in image_paths:
        if path.exists():
            images.append(Image.open(path))

    contents = [prompt] + images

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ConstraintSchema,
            temperature=0.0,
        ),
    )
    
    if response.parsed:
        return response.parsed
    else:
        raise ValueError("Failed to parse ConstraintSchema from VLM response.")
