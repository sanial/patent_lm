from __future__ import annotations

def schema_to_diffusion_prompt(constraint_schema: dict) -> str:
    """Convert a Gemma ``ConstraintSchema`` dict into a diffusion-style prompt.

    Picks up to three part names, two material hints, and two geometry
    hints from the schema and stitches them into an object-centric,
    studio-style prompt that works well with SF3D / Trellis style
    image-to-3D models.

    Args:
        constraint_schema: Dict produced by the VLM parser; should contain
            ``parts`` (list of part dicts), and optionally
            ``geometry_hints``.

    Returns:
        A single-line prompt string suitable for a text-to-image model.
    """
    parts = constraint_schema.get("parts", [])
    part_names = [p.get("name", p.get("primitive_type", "object")) for p in parts]
    material_hints = list(set(
        p.get("material_hint", "") for p in parts if p.get("material_hint")
    ))
    geometry_hints = constraint_schema.get("geometry_hints", [])

    components = ", ".join(part_names[:3]) if part_names else "mechanical part"
    materials = " and ".join(material_hints[:2]) if material_hints else "metal"
    geo = " ".join(geometry_hints[:2]) if geometry_hints else ""

    prompt = (
        f"a {materials} {components}, {geo} object, "
        "product photography, white studio background, "
        "soft directional lighting, sharp focus, photorealistic, "
        "no text, no labels, no annotations"
    )
    return prompt
