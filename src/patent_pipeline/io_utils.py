from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing and return its :class:`Path`.

    Args:
        path: Filesystem path to create.

    Returns:
        The same path as a :class:`pathlib.Path` after ``mkdir(parents=True,
        exist_ok=True)``.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts.

    Blank lines are skipped. Each non-empty line must decode to a JSON
    object.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed dict rows.

    Raises:
        ValueError: On invalid JSON or non-object rows; the line number is
            included in the error message.
    """
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write an iterable of dicts as a JSONL file.

    Creates parent directories as needed. Uses ``ensure_ascii=False`` so
    Unicode survives the round-trip.

    Args:
        path: Destination JSONL path.
        rows: Iterable of dicts to serialize, one per line.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
