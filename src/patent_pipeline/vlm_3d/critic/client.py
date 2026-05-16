import os
from pathlib import Path
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

from .schema import CriticSchema
from ..parser.schema import ConstraintSchema

def critique_render(
    rendered_view_paths: list[Path],
    constraints: ConstraintSchema,
    model_name: str = "gemini-1.5-pro",
) -> CriticSchema:
    
    api_key = os.environ.get("GEMINI_API_KEY", "")
    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are a 3D modeling critic.
    Review these rendered views of a generated 3D candidate against the following constraints:
    {constraints.model_dump_json()}
    
    Output a detailed critique scoring the view consistency, boundary fidelity, and any issues found.
    """
    
    images = []
    for path in rendered_view_paths:
        if path.exists():
            images.append(Image.open(path))

    contents = [prompt] + images

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CriticSchema,
            temperature=0.0,
        ),
    )
    
    if response.parsed:
        return response.parsed
    else:
        raise ValueError("Failed to parse CriticSchema from VLM response.")
