from __future__ import annotations

def schema_to_diffusion_prompt(constraint_schema: dict) -> str:
    """
    Convert Gemma ConstraintSchema to a diffusion prompt.
    Produces object-centric, studio-style prompt for best SF3D/Trellis compatibility.
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
