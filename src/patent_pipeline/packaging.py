from __future__ import annotations

import json
from pathlib import Path

from .io_utils import write_jsonl


def write_final_jsonl(records: list[dict], output_jsonl: str | Path) -> None:
    write_jsonl(output_jsonl, records)


def build_hf_dataset(records: list[dict], out_dir: str | Path) -> None:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("datasets package is required for build_hf_dataset=True") from exc

    normalized: list[dict] = []
    for row in records:
        normalized.append(
            {
                "patent_id": row.get("patent_id"),
                "image": row.get("views", {}).get("front") if isinstance(row.get("views"), dict) else None,
                "masks_path": row.get("masks_path"),
                "poses": json.dumps(row.get("poses", []), ensure_ascii=False),
                "shapes": json.dumps(row.get("shapes", []), ensure_ascii=False),
                "caption": row.get("caption"),
                "cpc": json.dumps(row.get("cpc", []), ensure_ascii=False),
            }
        )

    ds = Dataset.from_list(normalized)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
