from __future__ import annotations

import ast
import csv
from pathlib import Path

from .io_utils import write_jsonl


def _safe_list(value: str) -> list[str]:
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
    if class_field is None:
        return []
    raw = str(class_field).strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _infer_view_key(desc: str, fallback_idx: int) -> str:
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
