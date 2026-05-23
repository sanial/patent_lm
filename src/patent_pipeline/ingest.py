from __future__ import annotations

import ast
import csv
from pathlib import Path

from .io_utils import write_jsonl


def _safe_list(value: str) -> list[str]:
    """Parse a CSV cell containing a Python-literal list of strings.

    The IMPACT CSV stores list-valued columns (file_names, fig_desc, ...) as
    a stringified Python list, e.g. ``"['front', 'back']"``. This helper
    safely evaluates that string and returns the contained elements as
    ``list[str]``.

    Args:
        value: Raw cell content (may be ``None`` or empty).

    Returns:
        List of string items, or an empty list if the cell is empty or does
        not contain a valid list literal.
    """
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return []


def _class_to_list(class_field: str) -> list[str]:
    """Split a comma-separated CPC class field into a list of trimmed strings.

    Args:
        class_field: Raw CPC field from the CSV (e.g. ``"D 2728, D2840"``).

    Returns:
        List of non-empty class codes with surrounding whitespace removed.
    """
    if class_field is None:
        return []
    raw = str(class_field).strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _infer_view_key(desc: str, fallback_idx: int) -> str:
    """Map a figure-description string to a canonical view key.

    Inspects the description text for orientation keywords (front, rear,
    left, right, top, bottom, perspective, side) and returns the
    corresponding key. If no keyword is matched, falls back to
    ``view_<idx>`` so the row still gets a unique slot.

    Args:
        desc: Free-text figure description from the CSV.
        fallback_idx: Positional index of this figure, used to build a
            fallback key when no orientation is detectable.

    Returns:
        A canonical view key such as ``"front"``, ``"perspective"``, or
        ``"view_02"``.
    """
    text = desc.lower()
    if "front" in text:
        return "front"
    if "rear" in text or "back" in text:
        return "back"
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if "top" in text or "plan" in text:
        return "top"
    if "bottom" in text:
        return "bottom"
    if "perspective" in text or "isometric" in text:
        return "perspective"
    if "side" in text or "elevational" in text or "elevation" in text:
        return "side"
    return f"view_{fallback_idx:02d}"


def _build_views(base_dir: Path, file_names: list[str], fig_desc: list[str]) -> dict[str, str]:
    """Assemble the ``views`` dict for a single patent.

    Pairs each filename with the matching figure description, derives a
    canonical view key via :func:`_infer_view_key`, and stores the absolute
    POSIX path. Duplicate keys are disambiguated with an index suffix. If
    no ``front`` key is produced, the first filename is used as the front
    view so downstream stages always have something to read.

    Args:
        base_dir: Directory containing the patent's image files.
        file_names: Filenames listed in the CSV (e.g. ``D00000.TIF``).
        fig_desc: Per-file textual description list (parallel to
            ``file_names``).

    Returns:
        Mapping of view key to absolute POSIX path of the image file.
    """
    views: dict[str, str] = {}
    for i, file_name in enumerate(file_names):
        desc = fig_desc[i] if i < len(fig_desc) else ""
        key = _infer_view_key(desc, i)
        if key in views:
            key = f"{key}_{i:02d}"
        views[key] = str((base_dir / file_name).as_posix())

    if "front" not in views and file_names:
        views["front"] = str((base_dir / file_names[0]).as_posix())
    return views


def build_manifest_from_impact_csv(
    csv_path: str | Path,
    image_root: str | Path,
    output_jsonl: str | Path,
    max_samples: int = 0,
) -> int:
    """Convert the IMPACT sample CSV into the project's raw JSONL manifest.

    Reads each CSV row, resolves image paths under ``image_root``, builds a
    canonical ``views`` dict, and emits a JSONL record per patent. Rows are
    dropped if they lack a patent id, date, or any front-view image.

    Args:
        csv_path: Path to the IMPACT metadata CSV.
        image_root: Root directory containing the per-patent image folders
            (``USD0908314-20210126/`` etc.).
        output_jsonl: Path for the JSONL manifest to write.
        max_samples: If > 0, stop after collecting this many rows.

    Returns:
        The number of rows written to ``output_jsonl``.
    """
    csv_file = Path(csv_path)
    root = Path(image_root)

    rows_out: list[dict] = []
    with csv_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patent_id = str(row.get("id", "")).strip()
            date = str(row.get("date", "")).strip()
            if not patent_id or not date:
                continue

            folder_name = f"US{patent_id}-{date}"
            folder_path = root / folder_name

            file_names = _safe_list(str(row.get("file_names", "")))
            fig_desc = _safe_list(str(row.get("fig_desc", "")))
            views = _build_views(folder_path, file_names=file_names, fig_desc=fig_desc)
            if "front" not in views:
                continue

            caption = str(row.get("caption", "")).strip()
            if not caption:
                caption = str(row.get("title", "")).strip()

            out_row = {
                "patent_id": patent_id,
                "title": str(row.get("title", "")).strip(),
                "caption": caption,
                "claim": str(row.get("claim", "")).strip(),
                "date": date,
                "cpc": _class_to_list(str(row.get("class", ""))),
                "class_search": _safe_list(str(row.get("class_search", ""))),
                "inv_country": _safe_list(str(row.get("inv_country", ""))),
                "views": views,
            }
            rows_out.append(out_row)
            if max_samples > 0 and len(rows_out) >= max_samples:
                break

    write_jsonl(output_jsonl, rows_out)
    return len(rows_out)
