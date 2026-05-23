from __future__ import annotations

import json
from pathlib import Path

from .io_utils import write_jsonl


def write_final_jsonl(records: list[dict], output_jsonl: str | Path) -> None:
    """Write the final per-patent records to a JSONL file.

    Thin wrapper over :func:`patent_pipeline.io_utils.write_jsonl` used by
    the ``package`` stage so the CLI has a single named export.

    Args:
        records: List of fully-populated manifest rows.
        output_jsonl: Destination path.
    """
    write_jsonl(output_jsonl, records)


def build_hf_dataset(records: list[dict], out_dir: str | Path) -> None:
    """Persist the records as a Hugging Face Datasets directory.

    Flattens nested fields (``poses``, ``shapes``, ``cpc``) to JSON strings
    so the resulting Arrow table has a stable, scalar-typed schema suitable
    for training and sharing.

    Args:
        records: List of fully-populated manifest rows.
        out_dir: Directory in which ``data-*.arrow`` and metadata files
            will be saved via ``Dataset.save_to_disk``.

    Raises:
        RuntimeError: If the ``datasets`` package is not installed.
    """
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
