from __future__ import annotations
from pathlib import Path
from .augmentor.prompt_builder import schema_to_diffusion_prompt
from .augmentor.gemini_augmentor import augment_figure_with_gemini
from .augmentor.proxy_selector import select_best_proxy
from .reconstructor.sf3d_runner import run_sf3d_reconstruction


def prepare_art3d_job(
    record: dict,
    config: "Vlm3dConfig",
    work_dir: Path,
) -> tuple[dict, dict | None]:
    """Run the Gemini-only portion of the Art3D loop.

    Returns ``(updated_record, sf3d_job_or_None)`` where ``sf3d_job`` is a dict
    ``{"input": <best_proxy>, "output": <planned mesh path>, "patent_id": ...}``
    that the caller should hand to ``run_sf3d_batch`` later. If preparation
    failed, ``sf3d_job`` is ``None`` and ``updated_record['art3d_result']``
    explains why.
    """
    print(f"Preparing Art3D job for {record['patent_id']}")
    updated = dict(record)

    constraint_schema = updated.get("vlm_schema")
    if not constraint_schema:
        updated["art3d_result"] = "failed: no vlm_schema in record"
        return updated, None

    prompt = schema_to_diffusion_prompt(constraint_schema)

    cleaned_figure_path = updated.get("cleaned_figure_path")
    if not cleaned_figure_path or not Path(cleaned_figure_path).exists():
        front_view = updated.get("views", {}).get("front")
        if front_view and Path(front_view).exists():
            print(f"  [art3d] no cleaned_figure_path; falling back to views.front: {front_view}")
            cleaned_figure_path = front_view
        else:
            updated["art3d_result"] = "failed: no cleaned_figure_path and no front view"
            return updated, None

    proxy_images = augment_figure_with_gemini(
        cleaned_figure_path=cleaned_figure_path,
        diffusion_prompt=prompt,
        n_candidates=config.num_candidates,
    )
    if not proxy_images:
        updated["art3d_result"] = "failed: augmentor returned no images"
        return updated, None

    proxy_dir = work_dir / "proxies" / record["patent_id"]
    proxy_dir.mkdir(parents=True, exist_ok=True)
    proxy_paths = []
    for i, img in enumerate(proxy_images):
        path = proxy_dir / f"proxy_{i}.png"
        img.save(path)
        proxy_paths.append(str(path))
    updated["proxy_image_paths"] = proxy_paths

    source_figure_path = updated.get("views", {}).get("front")
    best_proxy_path = select_best_proxy(
        proxy_image_paths=proxy_paths,
        source_figure_path=source_figure_path,
        constraint_schema=constraint_schema,
    )
    updated["best_proxy_path"] = best_proxy_path

    mesh_output_dir = work_dir / "reconstructed_meshes"
    mesh_output_dir.mkdir(parents=True, exist_ok=True)
    planned_mesh_path = str(mesh_output_dir / f"{record['patent_id']}.glb")
    updated["art3d_result"] = "prepared"
    updated["planned_mesh_path"] = planned_mesh_path

    job = {
        "input": best_proxy_path,
        "output": planned_mesh_path,
        "patent_id": record["patent_id"],
    }
    return updated, job


def run_art3d_loop(
    record: dict,
    config: "Vlm3dConfig",
    work_dir: Path,
    device: str = "cuda",
) -> dict:
    """Single-record Art3D loop (Gemini + one SF3D subprocess).

    Kept for callers that want the legacy one-shot behaviour. The batched
    pipeline path in cli.py uses ``prepare_art3d_job`` + ``run_sf3d_batch``
    instead.
    """
    updated, job = prepare_art3d_job(record, config, work_dir)
    if job is None:
        return updated
    reconstructed_mesh_path = run_sf3d_reconstruction(
        proxy_image_path=job["input"],
        output_path=job["output"],
        device=device,
    )
    updated["art3d_result"] = "processed"
    updated["mesh_path"] = str(reconstructed_mesh_path)
    return updated
