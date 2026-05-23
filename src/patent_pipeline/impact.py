from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .io_utils import ensure_dir


def bootstrap_workspace(config_example: str | Path, config_target: str | Path, manifest_path: str | Path) -> None:
    """Initialize a fresh workspace with a config and a seed manifest.

    Copies the example YAML to the target path (if it does not already
    exist) and writes a single placeholder row to the manifest JSONL (also
    only if missing) so the downstream pipeline has something to chew on.

    Args:
        config_example: Source example YAML (typically
            ``configs/pipeline.example.yaml``).
        config_target: Destination YAML path (typically
            ``configs/pipeline.yaml``).
        manifest_path: Destination JSONL path for the seed manifest.
    """
    config_example_path = Path(config_example)
    config_target_path = Path(config_target)
    manifest_out = Path(manifest_path)

    config_target_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_target_path.exists():
        shutil.copyfile(config_example_path, config_target_path)

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_out.exists():
        sample = (
            '{"patent_id":"D0912345","cpc":["A47C3/00"],'
            '"caption":"Sample chair design.",'
            '"views":{"front":"data/images/D0912345/front.png",'
            '"side":"data/images/D0912345/side.png",'
            '"top":"data/images/D0912345/top.png",'
            '"perspective":"data/images/D0912345/perspective.png"}}\n'
        )
        manifest_out.write_text(sample, encoding="utf-8")


def clone_impact_repo(output_root: str | Path, force: bool = False) -> Path:
    """Clone (or re-clone) the upstream IMPACT repository.

    Args:
        output_root: Directory under which an ``IMPACT/`` clone will be
            created.
        force: If True and the target already exists, remove it first.

    Returns:
        Path to the cloned ``IMPACT`` directory.

    Raises:
        RuntimeError: When ``git clone`` exits non-zero; the first 1000
            chars of stderr (or stdout) are surfaced.
    """
    out_root = ensure_dir(output_root)
    repo_dir = out_root / "IMPACT"
    if repo_dir.exists():
        if force:
            shutil.rmtree(repo_dir)
        else:
            return repo_dir

    cmd = ["git", "clone", "https://github.com/AI4Patents/IMPACT.git", str(repo_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip()[:1000])
    return repo_dir
