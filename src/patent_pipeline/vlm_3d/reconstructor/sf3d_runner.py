"""Driver for the external Stable-Fast-3D venv.

Spawns the standalone runner script (`scripts/run_sf3d_external.py`) inside the
separate sf3d virtualenv. Supports single-image and batched-job modes; the
batched mode loads the SF3D model only once per call.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SF3D_PYTHON_EXECUTABLE = r"c:\Users\Sunny\OneDrive\Documents\patent-3d-viewer\external\stable-fast-3d\.venv\Scripts\python.exe"
SF3D_PROJECT_ROOT = r"c:\Users\Sunny\OneDrive\Documents\patent-3d-viewer\external\stable-fast-3d"
SF3D_RUNNER_SCRIPT = r"c:\Users\Sunny\OneDrive\Documents\patent_lm\scripts\run_sf3d_external.py"

logger = logging.getLogger(__name__)


def _build_env() -> dict:
    """Build the subprocess environment with SF3D project root on PYTHONPATH.

    Returns:
        A copy of ``os.environ`` with the SF3D project root prepended to
        ``PYTHONPATH`` and ``PYTHONUNBUFFERED=1`` set so the child's stdout
        streams live.
    """
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        SF3D_PROJECT_ROOT + (os.pathsep + existing_pp if existing_pp else "")
    )
    # Make stdout line-buffered in the child so we see progress in real time.
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _stream_subprocess(command: list[str]) -> tuple[int, str]:
    """Run a command, streaming combined stdout/stderr live to our stdout.

    Used so users see SF3D progress in real time instead of waiting for the
    subprocess to finish.

    Args:
        command: Argument vector to execute.

    Returns:
        Tuple ``(returncode, full_stdout)`` where ``full_stdout`` contains
        the concatenated output that was also echoed live.
    """
    print(f"[sf3d-runner] launching: {' '.join(command)}", flush=True)
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=SF3D_PROJECT_ROOT,
        env=_build_env(),
        bufsize=1,
    )
    collected: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        collected.append(line)
    rc = proc.wait()
    return rc, "".join(collected)


def run_sf3d_reconstruction(
    proxy_image_path: str,
    output_dir: str | Path | None = None,
    file_name: str | None = None,
    output_path: str | None = None,
    device: str = "cuda",
) -> str:
    """Run SF3D on a single image inside the external venv.

    Either ``output_path`` or both ``output_dir`` + ``file_name`` must be
    supplied. Parent directories are created on demand.

    Args:
        proxy_image_path: Path to the input image (proxy render).
        output_dir: Directory to write the resulting ``.glb`` into.
        file_name: Base filename (``.glb`` suffix added if missing).
        output_path: Explicit destination ``.glb`` path; overrides
            ``output_dir``/``file_name``.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        Absolute path to the resulting mesh.

    Raises:
        ValueError: If neither ``output_path`` nor the dir/name pair is
            provided.
        RuntimeError: If the SF3D subprocess exits non-zero.
        FileNotFoundError: If the subprocess succeeded but the expected
            mesh file is missing.
    """
    if output_path is None:
        if output_dir is None or file_name is None:
            raise ValueError(
                "run_sf3d_reconstruction: must provide either output_path, "
                "or both output_dir and file_name"
            )
        name = file_name if file_name.endswith(".glb") else f"{file_name}.glb"
        output_path = str(Path(output_dir) / name)

    abs_input = str(Path(proxy_image_path).resolve())
    abs_output = str(Path(output_path).resolve())
    Path(abs_output).parent.mkdir(parents=True, exist_ok=True)

    command = [
        SF3D_PYTHON_EXECUTABLE,
        SF3D_RUNNER_SCRIPT,
        "--input_image", abs_input,
        "--output_mesh", abs_output,
        "--device", device,
    ]
    rc, _ = _stream_subprocess(command)
    if rc != 0:
        raise RuntimeError(f"SF3D subprocess failed with exit code {rc}")
    if not Path(abs_output).exists():
        raise FileNotFoundError(f"SF3D finished but no mesh at {abs_output}")
    return abs_output


def run_sf3d_batch(
    jobs: list[dict],
    device: str = "cuda",
) -> list[dict]:
    """Run SF3D on many images in a *single* subprocess.

    Args:
        jobs: list of {"input": <image path>, "output": <.glb path>, ...extra}.
              Extra keys are preserved on the returned dicts.
        device: "cuda" or "cpu".

    Returns:
        A list (same order, same length) with each job dict augmented by
        ``{"ok": bool, "error": Optional[str], "output": <abs path>}``.
    """
    if not jobs:
        return []

    # Resolve to absolute paths and ensure output dirs exist.
    resolved_jobs: list[dict] = []
    for j in jobs:
        abs_in = str(Path(j["input"]).resolve())
        abs_out = str(Path(j["output"]).resolve())
        Path(abs_out).parent.mkdir(parents=True, exist_ok=True)
        resolved_jobs.append({**j, "input": abs_in, "output": abs_out})

    # Write the manifest to a temp file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        json.dump(
            [{"input": j["input"], "output": j["output"]} for j in resolved_jobs],
            tf,
        )
        manifest_path = tf.name

    try:
        command = [
            SF3D_PYTHON_EXECUTABLE,
            SF3D_RUNNER_SCRIPT,
            "--jobs", manifest_path,
            "--device", device,
        ]
        rc, full_stdout = _stream_subprocess(command)
        if rc != 0:
            raise RuntimeError(f"SF3D batch subprocess failed with exit code {rc}")

        # Parse the trailing RESULTS=... line emitted by the runner.
        results: list[dict] | None = None
        for line in reversed(full_stdout.splitlines()):
            m = re.search(r"\[sf3d-ext\]\s+RESULTS=(.*)$", line)
            if m:
                try:
                    results = json.loads(m.group(1))
                except json.JSONDecodeError:
                    results = None
                break
        if results is None:
            # Fall back to checking file existence per job.
            results = [
                {"input": j["input"], "output": j["output"], "ok": Path(j["output"]).exists()}
                for j in resolved_jobs
            ]

        # Index results by input path so we can merge back into the original
        # jobs (preserving any extra keys the caller attached).
        by_input = {r["input"]: r for r in results}
        merged: list[dict] = []
        for j in resolved_jobs:
            r = by_input.get(j["input"], {"ok": False, "error": "no result reported"})
            merged.append({**j, **r})
        return merged
    finally:
        try:
            os.unlink(manifest_path)
        except OSError:
            pass
