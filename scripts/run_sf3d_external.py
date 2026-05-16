"""Standalone runner executed inside the external stable-fast-3d venv.

Supports two modes:

  1. Single-image:
       python run_sf3d_external.py --input_image PATH --output_mesh PATH [--device cuda]

  2. Batched (model loads once for many images):
       python run_sf3d_external.py --jobs JOBS.json [--device cuda]
     where JOBS.json is a JSON array of {"input": "...", "output": "..."}.

In batched mode the script prints a final line:
       [sf3d-ext] RESULTS=<json>
that the caller parses to get per-job ok/error status.
"""
import argparse
import json
import os
import sys
import traceback
from contextlib import nullcontext

import torch
from PIL import Image


def _load_model(device: str):
    import rembg
    from sf3d.system import SF3D

    print(f"[sf3d-ext] device={device}", flush=True)
    print("[sf3d-ext] loading model stabilityai/stable-fast-3d ...", flush=True)
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.to(device)
    model.eval()
    rembg_session = rembg.new_session()
    return model, rembg_session


def _prepare_image(path: str, rembg_session):
    from sf3d.utils import remove_background, resize_foreground

    img = Image.open(path).convert("RGBA")
    img = remove_background(img, rembg_session)
    img = resize_foreground(img, 0.85)
    return img


def _reconstruct(model, image, device: str, output_path: str) -> None:
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.startswith("cuda")
        else nullcontext()
    )
    with torch.no_grad(), autocast_ctx:
        mesh, _ = model.run_image(
            [image],
            bake_resolution=1024,
            remesh="none",
            vertex_count=-1,
        )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mesh.export(output_path, include_normals=True)


def run_single(input_image: str, output_mesh: str, device: str) -> int:
    if not os.path.exists(input_image):
        print(f"[sf3d-ext] ERROR: input not found: {input_image}", file=sys.stderr)
        return 2
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model, rembg_session = _load_model(device)
    image = _prepare_image(input_image, rembg_session)
    print(f"[sf3d-ext] reconstructing -> {output_mesh}", flush=True)
    _reconstruct(model, image, device, output_mesh)
    print("[sf3d-ext] done", flush=True)
    return 0


def run_batch(jobs_path: str, device: str) -> int:
    with open(jobs_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    if not isinstance(jobs, list) or not jobs:
        print("[sf3d-ext] ERROR: empty jobs manifest", file=sys.stderr)
        return 2

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model, rembg_session = _load_model(device)

    results = []
    total = len(jobs)
    for i, job in enumerate(jobs):
        inp = job["input"]
        out = job["output"]
        print(f"[sf3d-ext] job {i + 1}/{total}: {inp}", flush=True)
        if not os.path.exists(inp):
            results.append({"input": inp, "output": out, "ok": False, "error": "input not found"})
            continue
        try:
            image = _prepare_image(inp, rembg_session)
            _reconstruct(model, image, device, out)
            ok = os.path.exists(out)
            results.append({
                "input": inp,
                "output": out,
                "ok": ok,
                "error": None if ok else "exported but file missing",
            })
            print(f"[sf3d-ext] job {i + 1}/{total} done -> {out}", flush=True)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc(limit=3)
            print(f"[sf3d-ext] job {i + 1}/{total} FAILED: {exc}\n{tb}", flush=True)
            results.append({"input": inp, "output": out, "ok": False, "error": str(exc)})
        finally:
            # Free transient tensors between jobs to keep VRAM stable.
            if device.startswith("cuda"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    print(f"[sf3d-ext] RESULTS={json.dumps(results)}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SF3D reconstruction.")
    parser.add_argument("--input_image")
    parser.add_argument("--output_mesh")
    parser.add_argument("--jobs", help="Path to JSON manifest for batched mode")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.jobs:
        return run_batch(args.jobs, args.device)
    if args.input_image and args.output_mesh:
        return run_single(args.input_image, args.output_mesh, args.device)
    parser.error("must provide either --jobs, or both --input_image and --output_mesh")
    return 2


if __name__ == "__main__":
    sys.exit(main())
