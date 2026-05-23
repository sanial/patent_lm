from __future__ import annotations

SURFACE_PATTERN_CPCS = {"D05", "D06", "D03", "D32"}
SCHEMATIC_CPCS        = {"G06", "H01", "H04", "H05"}
ASSEMBLY_CPCS         = {"F", "B", "E"}

def route_patent(cpc_list: list[str]) -> str:
    """Route a patent to a downstream processing pipeline based on CPC codes.

    Inspects the first three normalized characters of each CPC and matches
    them against the surface-pattern, schematic, and assembly prefix sets.
    The first match wins; rows with no CPCs or no recognised prefix fall
    through to ``"object"`` (the default object-reconstruction pipeline).

    Args:
        cpc_list: List of CPC code strings from the manifest row.

    Returns:
        One of ``"surface_pattern"``, ``"schematic"``, ``"assembly"`` or
        ``"object"``.
    """
    if not cpc_list:
        return "object"  # Default if no CPC codes are provided

    for cpc in cpc_list:
        # Normalize by removing spaces and taking the first 3 chars of the main class
        prefix = cpc.strip().replace(" ", "")[:3]
        if any(prefix.startswith(k) for k in SURFACE_PATTERN_CPCS):
            return "surface_pattern"
        if any(prefix.startswith(k) for k in SCHEMATIC_CPCS):
            return "schematic"
        if any(prefix.startswith(k) for k in ASSEMBLY_CPCS):
            return "assembly"
            
    return "object"  # Default for patents like furniture (A47) that don't fit other categories
