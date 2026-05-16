from __future__ import annotations

import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from tqdm import tqdm

from .config import PipelineConfig, load_config
from .filtering import filter_records
from .io_utils import ensure_dir, read_jsonl, write_jsonl
from .impact import bootstrap_workspace, clone_impact_repo
from .ingest import build_manifest_from_impact_csv
from .masks import build_masks_for_record
from .packaging import build_hf_dataset, write_final_jsonl
from .shape_index import build_proxy_shape_index_from_manifest
from .shapes import prewarm_clip_resources
from .vlm_3d.loop import run_art3d_loop, prepare_art3d_job
from .vlm_3d.reconstructor.sf3d_runner import run_sf3d_batch
from .vlm_3d.surface.loop import run_surface_pattern_loop
from .routing import route_patent


def _stage_paths(cfg: PipelineConfig) -> dict[str, Path]:
    work_dir = ensure_dir(cfg.paths.work_dir)
    return {
        "filtered": work_dir / "filtered_manifest.jsonl",
        "masked": work_dir / "masked_manifest.jsonl",
        "posed": work_dir / "posed_manifest.jsonl",
        "shaped": work_dir / "shaped_manifest.jsonl",
    }


def _run_with_retries(func, row: dict, retries: int) -> dict:
    attempts = max(retries, 1)
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return func(row)
        except Exception as exc:  # pragma: no cover - runtime resilience path
            last_exc = exc
    if last_exc is None:
        raise RuntimeError("Unknown retry failure")
    raise last_exc


def _process_rows_with_failures(
    rows: list[dict],
    stage_name: str,
    retries: int,
    workers: int,
    fn,
) -> tuple[list[dict], list[dict]]:
    if workers <= 1:
        ok_rows: list[dict] = []
        failures: list[dict] = []
        for row in tqdm(rows, desc=stage_name):
            patent_id = str(row.get("patent_id", "unknown"))
            try:
                ok_rows.append(_run_with_retries(fn, row, retries=retries))
            except Exception:
                failures.append(
                    {
                        "stage": stage_name,
                        "patent_id": patent_id,
                        "error": traceback.format_exc(limit=1).strip(),
                        "traceback": traceback.format_exc(limit=2),
                    }
                )
        return ok_rows, failures

    def _worker(idx: int, row: dict) -> tuple[int, dict | None, dict | None]:
        patent_id = str(row.get("patent_id", "unknown"))
        try:
            result = _run_with_retries(fn, row, retries=retries)
            return idx, result, None
        except Exception as exc:
            failure = {
                "stage": stage_name,
                "patent_id": patent_id,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=2),
            }
            return idx, None, failure

    ok_rows: list[dict] = []
    failures: list[dict] = []
    indexed_ok: list[tuple[int, dict]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker, i, row) for i, row in enumerate(rows)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=stage_name):
            idx, ok_row, failure = fut.result()
            if ok_row is not None:
                indexed_ok.append((idx, ok_row))
            if failure is not None:
                failures.append(failure)

    indexed_ok.sort(key=lambda x: x[0])
    ok_rows = [row for _, row in indexed_ok]
    return ok_rows, failures


def run_filter(cfg: PipelineConfig) -> Path:
    stage = _stage_paths(cfg)
    rows = read_jsonl(cfg.paths.raw_manifest)
    filtered = filter_records(rows, cfg.filter.target_cpc_prefixes, cfg.filter.max_samples)
    write_jsonl(stage["filtered"], filtered)
    print(f"[filter] kept {len(filtered)} rows -> {stage['filtered']}")
    return stage["filtered"]


def run_masks(cfg: PipelineConfig) -> Path:
    stage = _stage_paths(cfg)
    rows = read_jsonl(stage["filtered"])
    masks_dir = ensure_dir(Path(cfg.paths.work_dir) / "masks")

    out, failures = _process_rows_with_failures(
        rows,
        "masks",
        retries=cfg.runtime.max_retries,
        workers=cfg.masks.workers,
        fn=lambda row: build_masks_for_record(
            row,
            masks_dir=masks_dir,
            min_component_area=cfg.masks.min_component_area,
        )
    )

    write_jsonl(stage["masked"], out)
    if failures:
        write_jsonl(Path(cfg.paths.work_dir) / "failures_masks.jsonl", failures)
    print(f"[masks] wrote {len(out)} rows -> {stage['masked']}")
    if failures:
        print(f"[masks] failures: {len(failures)}")
    return stage["masked"]


def run_poses(cfg: PipelineConfig) -> Path:
    stage = _stage_paths(cfg)
    rows = read_jsonl(stage["masked"])
    
    out, failures = _process_rows_with_failures(
        rows,
        "poses_parser",
        retries=cfg.runtime.max_retries,
        workers=cfg.pose.workers,
        fn=lambda row: _run_vlm_parser(row, cfg.vlm_3d.model_name)
    )

    write_jsonl(stage["posed"], out)
    if failures:
        write_jsonl(Path(cfg.paths.work_dir) / "failures_poses.jsonl", failures)
    print(f"[poses] parsed {len(out)} rows via VLM -> {stage['posed']}")
    if failures:
        print(f"[poses] failures: {len(failures)}")
    return stage["posed"]

def _run_vlm_parser(record: dict, model_name: str) -> dict:
    updated = dict(record)
    views = updated.get("views", {})
    front_path = Path(str(views.get("front", "")))
    caption = updated.get("caption", "")
    
    if front_path.exists():
        schema = parse_constraints([front_path], caption, model_name)
        updated["vlm_schema"] = schema.model_dump()
    return updated


def run_shapes(cfg: PipelineConfig, mode: str, limit: int = 0, patent_id: str | None = None) -> Path:
    stage = _stage_paths(cfg)
    rows = read_jsonl(stage["posed"])
    if patent_id:
        wanted = {p.strip() for p in patent_id.split(",") if p.strip()}
        rows = [r for r in rows if str(r.get("patent_id")) in wanted]
        print(f"[shapes] filtered to patent_id(s)={sorted(wanted)} ({len(rows)} rows)")
    if limit and limit > 0:
        rows = rows[:limit]
        print(f"[shapes] limited to first {len(rows)} records")
    work_dir = Path(cfg.paths.work_dir) / "vlm_3d"

    # prewarm CLIP only for the legacy optimize mode — art3d doesn't use the
    # proxy-shape matching system and loading sentence-transformers here has
    # been triggering a native (pyarrow) access violation crash.
    if mode == "optimize":
        prewarm_clip_resources(
            emb_path=cfg.shape.shapenet_embeddings_npy,
            mesh_index_path=cfg.shape.shapenet_mesh_index_json,
            clip_model_name=cfg.shape.clip_model_name,
        )

    if mode == "art3d":
        return _run_shapes_art3d_batched(cfg, rows, work_dir, stage)

    # ---- legacy / optimize mode ----
    from .vlm_3d.legacy_optimize.loop import run_optimization
    fn = lambda row: run_optimization(
        row,
        out_dir=ensure_dir(Path(cfg.paths.work_dir) / "renders"),
        max_samples=cfg.runtime.max_samples,
    )
    out, failures = _process_rows_with_failures(
        rows,
        "shapes_optimize",
        retries=cfg.runtime.max_retries,
        workers=cfg.shape.workers,
        fn=fn,
    )

    write_jsonl(stage["shaped"], out)
    if failures:
        write_jsonl(Path(cfg.paths.work_dir) / "failures_shapes.jsonl", failures)
    print(f"[shapes] optimized {len(out)} rows -> {stage['shaped']}")
    if failures:
        print(f"[shapes] failures: {len(failures)}")
    return stage["shaped"]


def _run_shapes_art3d_batched(
    cfg: PipelineConfig,
    rows: list[dict],
    work_dir: Path,
    stage: dict[str, Path],
) -> Path:
    """Two-phase art3d:
      Phase A — Gemini augment + proxy select per record (no CUDA in main process).
      Phase B — single batched SF3D subprocess that loads the model once.
    """
    import os as _os

    # Keep CUDA out of the main process. Only the SF3D subprocess (which has
    # its own venv) should ever talk to the GPU. This prevents the main
    # python from creating a competing CUDA context that hangs the driver.
    _os.environ["CUDA_VISIBLE_DEVICES"] = ""

    prepared_rows: list[dict] = []
    jobs: list[dict] = []

    # ---- Phase A: per-record Gemini work (serial) ----
    for row in tqdm(rows, desc="art3d_prepare"):
        try:
            patent_type = route_patent(row.get("cpc", []))
            if patent_type == "surface_pattern":
                # Surface-pattern records aren't routed through SF3D; keep
                # their existing loop.
                prepared_rows.append(run_surface_pattern_loop(row, cfg.vlm_3d, work_dir))
                continue
            updated, job = prepare_art3d_job(row, cfg.vlm_3d, work_dir)
            prepared_rows.append(updated)
            if job is not None:
                jobs.append(job)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[art3d_prepare] {row.get('patent_id')} failed: {exc}\n{tb}")
            failed = dict(row)
            failed["art3d_result"] = f"prepare_failed: {exc}"
            prepared_rows.append(failed)

    # ---- Phase B: one batched SF3D subprocess for everything ----
    if jobs:
        print(f"[shapes/art3d] reconstructing {len(jobs)} meshes via batched SF3D ...")
        results = run_sf3d_batch(jobs, device="cuda")
        # Index results by patent_id (we attached it in prepare_art3d_job).
        by_pid = {r.get("patent_id"): r for r in results}
        for row in prepared_rows:
            r = by_pid.get(row.get("patent_id"))
            if r is None:
                continue
            if r.get("ok") and Path(r["output"]).exists():
                row["mesh_path"] = r["output"]
                row["art3d_result"] = "processed"
            else:
                row["art3d_result"] = f"sf3d_failed: {r.get('error', 'no output')}"
    else:
        print("[shapes/art3d] no SF3D jobs to run")

    write_jsonl(stage["shaped"], prepared_rows)
    failures = [r for r in prepared_rows if str(r.get("art3d_result", "")).startswith(("failed", "prepare_failed", "sf3d_failed"))]
    if failures:
        write_jsonl(Path(cfg.paths.work_dir) / "failures_shapes.jsonl", failures)
        print(f"[shapes/art3d] failures: {len(failures)}")
    print(f"[shapes/art3d] wrote {len(prepared_rows)} rows -> {stage['shaped']}")
    return stage["shaped"]

def _run_legacy_vlm_loop(record: dict, cfg: PipelineConfig, work_dir: Path) -> dict:
    from .vlm_3d.parser.schema import ConstraintSchema
    schema_dict = record.get("vlm_schema")
    if not schema_dict:
        return record
        
    schema = ConstraintSchema(**schema_dict)
    return run_optimization(record, schema, cfg.vlm_3d, work_dir)

def _run_vlm_loop(record: dict, cfg: PipelineConfig, work_dir: Path) -> dict:
    print(f"[_run_vlm_loop] Processing record: {record.get('patent_id')}")
    patent_type = route_patent(record.get("cpc", []))
    print(f"[_run_vlm_loop] Routed to: {patent_type}")
    
    if patent_type == "surface_pattern":
        print("[_run_vlm_loop] Running surface pattern loop...")
        return run_surface_pattern_loop(record, cfg.vlm_3d, work_dir)
    
    # Default to art3d loop for object, assembly, schematic
    print("[_run_vlm_loop] Running Art3D loop...")
    result = run_art3d_loop(record, cfg.vlm_3d, work_dir)
    print("[_run_vlm_loop] Art3D loop finished.")
    return result


def run_package(cfg: PipelineConfig) -> Path:
    stage = _stage_paths(cfg)
    rows = read_jsonl(stage["shaped"])

    output_jsonl = Path(cfg.packaging.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_final_jsonl(rows, output_jsonl)
    print(f"[package] wrote final jsonl -> {output_jsonl}")

    if cfg.packaging.build_hf_dataset:
        build_hf_dataset(rows, cfg.packaging.hf_dataset_dir)
        print(f"[package] wrote HF dataset -> {cfg.packaging.hf_dataset_dir}")

    return output_jsonl


def run_all(cfg: PipelineConfig, mode: str) -> None:
    run_filter(cfg)
    run_masks(cfg)
    run_poses(cfg)
    run_shapes(cfg, mode)
    run_package(cfg)


def run_bootstrap() -> None:
    bootstrap_workspace(
        config_example="configs/pipeline.example.yaml",
        config_target="configs/pipeline.yaml",
        manifest_path="data/impact_manifest.jsonl",
    )
    print("[bootstrap] created configs/pipeline.yaml and data/impact_manifest.jsonl if missing")


def run_download_impact(target_dir: str, force: bool) -> None:
    repo_dir = clone_impact_repo(target_dir, force=force)
    print(f"[download-impact] cloned IMPACT repo -> {repo_dir}")


def run_ingest_impact(csv_path: str, image_root: str, output_manifest: str, max_samples: int) -> None:
    count = build_manifest_from_impact_csv(
        csv_path=csv_path,
        image_root=image_root,
        output_jsonl=output_manifest,
        max_samples=max_samples,
    )
    print(f"[ingest-impact] wrote {count} rows -> {output_manifest}")


def run_build_proxy_shape_index(
    manifest_jsonl: str,
    output_embeddings_npy: str,
    output_mesh_index_json: str,
    clip_model_name: str,
    max_meshes: int,
) -> None:
    count = build_proxy_shape_index_from_manifest(
        manifest_jsonl=manifest_jsonl,
        output_embeddings_npy=output_embeddings_npy,
        output_mesh_index_json=output_mesh_index_json,
        clip_model_name=clip_model_name,
        max_meshes=max_meshes,
    )
    print(f"[build-proxy-shape-index] wrote {count} proxy mesh entries")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patent 2D->3D weak supervision dataset builder")
    parser.add_argument(
        "command",
        choices=[
            "bootstrap",
            "download-impact",
            "ingest-impact",
            "build-proxy-shape-index",
            "filter",
            "masks",
            "poses",
            "shapes",
            "package",
            "all",
        ],
    )
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to YAML config")
    parser.add_argument("--impact-dir", default="data/external", help="Directory where IMPACT repo is cloned")
    parser.add_argument(
        "--impact-csv",
        default="data/external/IMPACT/Sample data/sample_data.csv",
        help="Path to IMPACT metadata CSV",
    )
    parser.add_argument(
        "--impact-image-root",
        default="data/external/IMPACT/Sample data",
        help="Root directory containing per-patent IMPACT image folders",
    )
    parser.add_argument(
        "--output-manifest",
        default="data/impact_manifest.jsonl",
        help="Output manifest path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum rows to ingest from IMPACT CSV (0 means all)",
    )
    parser.add_argument(
        "--manifest-jsonl",
        default="data/impact_manifest.jsonl",
        help="Manifest used to derive proxy shape labels",
    )
    parser.add_argument(
        "--mode",
        default="art3d",
        choices=["art3d", "optimize"],
        help="Shape generation mode to run."
    )
    parser.add_argument(
        "--shape-embeddings-npy",
        default="data/shapenet_proxy/shapenet_embeddings.npy",
        help="Output path for proxy shape embeddings .npy",
    )
    parser.add_argument(
        "--shape-mesh-index-json",
        default="data/shapenet_proxy/shapenet_mesh_index.json",
        help="Output path for proxy mesh index JSON",
    )
    parser.add_argument(
        "--max-meshes",
        type=int,
        default=0,
        help="Maximum number of proxy mesh labels to include (0 means all)",
    )
    parser.add_argument("--force", action="store_true", help="Force re-clone IMPACT if target exists")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of records processed (0 = all)")
    parser.add_argument("--patent-id", default=None, help="Process only the given patent_id(s). Comma-separated for multiple.")
    return parser


def main() -> None:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "bootstrap":
        run_bootstrap()
        return

    if args.command == "download-impact":
        run_download_impact(args.impact_dir, force=args.force)
        return

    if args.command == "ingest-impact":
        run_ingest_impact(
            csv_path=args.impact_csv,
            image_root=args.impact_image_root,
            output_manifest=args.output_manifest,
            max_samples=args.max_samples,
        )
        return

    if args.command == "build-proxy-shape-index":
        cfg = load_config(args.config)
        run_build_proxy_shape_index(
            manifest_jsonl=args.manifest_jsonl,
            output_embeddings_npy=args.shape_embeddings_npy,
            output_mesh_index_json=args.shape_mesh_index_json,
            clip_model_name=cfg.shape.clip_model_name,
            max_meshes=args.max_meshes,
        )
        return

    cfg = load_config(args.config)

    if args.command == "filter":
        run_filter(cfg)
    elif args.command == "masks":
        run_masks(cfg)
    elif args.command == "poses":
        run_poses(cfg)
    elif args.command == "shapes":
        run_shapes(cfg, args.mode, limit=args.limit, patent_id=args.patent_id)
    elif args.command == "package":
        run_package(cfg)
    else:
        run_all(cfg, args.mode)


if __name__ == "__main__":
    main()
