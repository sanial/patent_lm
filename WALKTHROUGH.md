# Patent → 3D Pipeline Walkthrough

This document explains what the pipeline does end-to-end, how it is wired
together, and the key decisions that made the Art3D path actually work on a
Windows + single-GPU workstation.

The repo turns patent line-art figures (from the **IMPACT** dataset) into 3D
mesh supervision (`.glb` files) plus a structured JSONL manifest suitable for
training a 2D→3D model.

---

## 1. High-level flow

```
impact_manifest.jsonl
        │
        ▼
 ┌──────────────┐
 │   filter     │  CPC / sample-count filter
 └──────────────┘
        │
        ▼
 ┌──────────────┐
 │    masks     │  EasyOCR + connected-component masks
 └──────────────┘
        │
        ▼
 ┌──────────────┐
 │    poses     │  VLM parser → vlm_schema (constraint JSON)
 └──────────────┘
        │
        ▼
 ┌──────────────┐
 │    shapes    │  ART3D mode: Gemini + SF3D
 └──────────────┘
        │
        ▼
 ┌──────────────┐
 │   package    │  patent_3d_supervision.jsonl + HF dataset
 └──────────────┘
```

Each stage reads/writes a stage manifest under `data/work/`:

| Stage    | Output                                        |
|----------|-----------------------------------------------|
| filter   | `data/work/filtered_manifest.jsonl`           |
| masks    | `data/work/masked_manifest.jsonl`             |
| poses    | `data/work/posed_manifest.jsonl`              |
| shapes   | `data/work/shaped_manifest.jsonl`             |
| package  | `data/output/patent_3d_supervision.jsonl`     |

The CLI entry point is `python -m src.patent_pipeline.cli <command>`
(`src/patent_pipeline/cli.py`).

---

## 2. Inputs and configuration

- **Input data**: IMPACT line-art figures, ingested into
  `data/impact_manifest.jsonl` (one JSON record per patent, with `patent_id`,
  `views.front`, `caption`, `cpc`, …).
- **Config**: `configs/pipeline.yaml` — controls paths, worker counts, models,
  and the Art3D candidate count (`vlm_3d.num_candidates`).
- **Secrets**: `.env` with `GEMINI_API_KEY=...` (loaded via `python-dotenv`).
- **External SF3D venv**: Stable-Fast-3D is reused from a sibling project
  rather than installed into this venv. Paths are hardcoded in
  `src/patent_pipeline/vlm_3d/reconstructor/sf3d_runner.py`:

  ```
  SF3D_PYTHON_EXECUTABLE = .../patent-3d-viewer/external/stable-fast-3d/.venv/Scripts/python.exe
  SF3D_PROJECT_ROOT      = .../patent-3d-viewer/external/stable-fast-3d
  SF3D_RUNNER_SCRIPT     = .../patent_lm/scripts/run_sf3d_external.py
  ```

---

## 3. Stages in detail

### 3.1 `filter`
`src/patent_pipeline/filtering.py` — keeps rows whose CPC starts with one of
`filter.target_cpc_prefixes` (empty list = pass-through), capped at
`filter.max_samples`.

### 3.2 `masks`
`src/patent_pipeline/masks.py` — uses EasyOCR to detect figure labels/text and
removes them, then builds a binary mask via connected-component analysis with a
minimum component area. Mask `.npz` files land under `data/work/masks/`.

### 3.3 `poses`
`src/patent_pipeline/cli.py::_run_vlm_parser` calls the VLM constraint parser
(`vlm_3d/parser/`) which uses Gemini 1.5 Pro to produce a structured
`ConstraintSchema` JSON describing the object (shape primitives, symmetry,
counts, etc.). The schema is attached to each record as `vlm_schema`.

> Note: the parser path is decoupled from the legacy CLIP retrieval system.
> The `parse_constraints` import is lazy — `shapes` does **not** require it.

### 3.4 `shapes` — Art3D mode (the main work)

This is the part that builds the actual 3D meshes. Two modes are supported:

- `--mode art3d` (default): Gemini image-to-image + Stable-Fast-3D.
- `--mode optimize`: legacy CLIP-retrieval + differentiable rendering
  (kept for reference; not used in the current run).

The Art3D pipeline is implemented as a **two-phase batched run**:

#### Phase A — Per-record Gemini work (no GPU)

For every input record we:

1. Build a diffusion prompt from `vlm_schema` →
   `vlm_3d/augmentor/prompt_builder.py`.
2. Generate **N photorealistic proxy candidates** from the line-art with
   `gemini-2.5-flash-image` →
   `vlm_3d/augmentor/gemini_augmentor.py::augment_figure_with_gemini`.
3. Save them to `data/work/vlm_3d/proxies/<patent_id>/proxy_<i>.png`
   (the user's original input image is **never overwritten**).
4. Ask `gemini-2.5-flash` to pick the best candidate that matches the source
   figure → `vlm_3d/augmentor/proxy_selector.py::select_best_proxy`.
5. Plan an SF3D job: `{input: best_proxy.png, output: <patent_id>.glb}`.

Implementation: `vlm_3d/loop.py::prepare_art3d_job` returns
`(updated_record, sf3d_job_or_None)` without touching CUDA.

#### Phase B — One batched SF3D subprocess (GPU)

After all records are prepared, the orchestrator fires a **single subprocess**
in the external SF3D venv that:

1. Loads `stabilityai/stable-fast-3d` and a `rembg` session **once**.
2. Iterates over all jobs in the manifest, running background-removal +
   reconstruction + `.glb` export per image.
3. Frees VRAM between jobs with `torch.cuda.empty_cache()`.
4. Prints a final `[sf3d-ext] RESULTS=<json>` line the parent parses to merge
   per-job ok/error back into the records.

Implementation:
- Driver: `vlm_3d/reconstructor/sf3d_runner.py::run_sf3d_batch`
  (streams stdout via `_stream_subprocess`, sets `PYTHONPATH` to the external
  project root, `CUDA_VISIBLE_DEVICES` unset in the child).
- Subprocess script: `scripts/run_sf3d_external.py`
  (accepts either `--input_image/--output_mesh` for one-shot, or
  `--jobs JOBS.json` for batched mode).

### 3.5 `package`
`src/patent_pipeline/packaging.py` writes the final per-record JSONL to
`data/output/patent_3d_supervision.jsonl` and optionally a Hugging Face
dataset under `data/output/hf_dataset/`.

---

## 4. Key design decisions and why

### 4.1 GPU isolation
The main Python process sets `CUDA_VISIBLE_DEVICES=""` for Art3D mode. Only
the SF3D subprocess sees the GPU. This prevents two competing CUDA contexts on
Windows, which was hanging the WDDM driver (and the whole desktop).

### 4.2 Batched single-subprocess SF3D
The pre-batched version spawned 44 subprocesses in a row, each cold-loading
SF3D + rembg from disk → GPU. Disk + VRAM churn caused the freezes. The
batched runner loads everything once and processes the job list in-process.

### 4.3 Streaming subprocess output
`subprocess.run(..., capture_output=True)` buffers all child output in RAM.
With chatty tqdm + load logs across 44 jobs that's significant. The new
`_stream_subprocess` uses `Popen` + line-buffered stdout and forwards it to
the parent in real time.

### 4.4 Skip CLIP prewarm in Art3D mode
`prewarm_clip_resources()` pulls in `sentence_transformers → datasets →
pyarrow`, and pyarrow's native DLL was access-violating on this Windows box.
We only prewarm CLIP when `--mode optimize`, since Art3D doesn't use it.

### 4.5 New `google-genai` SDK
Both the augmentor and the proxy selector were rewritten to use
`google-genai` (`genai.Client(api_key=...).models.generate_content(...)`) —
the old `google.generativeai` (`genai.configure` + `GenerativeModel`) API
does not work with the current dependency set. Image data is read from
`candidate.content.parts[].inline_data.data`.

### 4.6 Preserve user input images
A user preference: never overwrite or redraw the originally provided figure.
The loop falls back to `views.front` when `cleaned_figure_path` is missing,
and proxies are written to a sibling folder.

### 4.7 Conditional / lazy legacy imports
The legacy CLIP-retrieval + differentiable-rendering path is loaded only
when `--mode optimize`. This keeps the Art3D run free of the older
shapenet/pytorch3d-heavy dependency chain.

### 4.8 `--patent-id` and `--limit` flags
Added to make iterative testing cheap: `--limit N` processes the first N
records, `--patent-id A,B,C` filters to specific IDs (any subset, any order).

---

## 5. How to run

### Prereqs

- This project's venv with `pip install -r requirements.txt` (no `sf3d`,
  no `pytorch3d` — those are intentionally not installed here).
- The external SF3D venv at the path baked into `sf3d_runner.py`.
- A `.env` containing `GEMINI_API_KEY=...`.

### Single record (recommended first run)

```powershell
python -m src.patent_pipeline.cli shapes --mode art3d --patent-id D0937859
```

### A subset

```powershell
python -m src.patent_pipeline.cli shapes --mode art3d --patent-id D0937859,D0908314
```

### First N records

```powershell
python -m src.patent_pipeline.cli shapes --mode art3d --limit 5
```

### All records, with crash trapping

```powershell
$env:PYTHONFAULTHANDLER=1
cmd /c "python -X faulthandler -u -m src.patent_pipeline.cli shapes --mode art3d > debug_out.txt 2>&1"
```

### Full end-to-end

```powershell
python -m src.patent_pipeline.cli all --mode art3d
```

---

## 6. Outputs

Per run, the following land in the workspace:

| Artifact             | Path                                                          |
|----------------------|---------------------------------------------------------------|
| Gemini proxies       | `data/work/vlm_3d/proxies/<patent_id>/proxy_<i>.png`          |
| Reconstructed mesh   | `data/work/vlm_3d/reconstructed_meshes/<patent_id>.glb`       |
| Shapes stage manifest| `data/work/shaped_manifest.jsonl`                             |
| Final supervision    | `data/output/patent_3d_supervision.jsonl`                     |
| HF dataset (opt.)    | `data/output/hf_dataset/`                                     |

Each row in `shaped_manifest.jsonl` is the input row plus:

```json
{
  "proxy_image_paths": ["data/work/vlm_3d/proxies/D0937859/proxy_0.png", "..."],
  "best_proxy_path":   "data/work/vlm_3d/proxies/D0937859/proxy_1.png",
  "planned_mesh_path": "data/work/vlm_3d/reconstructed_meshes/D0937859.glb",
  "mesh_path":         "data/work/vlm_3d/reconstructed_meshes/D0937859.glb",
  "art3d_result":      "processed"
}
```

`art3d_result` values:

- `processed` — full success.
- `prepared` — Gemini stage OK but SF3D had not yet run (transient).
- `failed: ...` — Phase A error.
- `sf3d_failed: ...` — Phase B error reported by the SF3D subprocess.

---

## 7. Known limitations / follow-ups

- `_run_vlm_parser` still references `parse_constraints` at call time. The
  top-level import was removed; if you re-run `poses`, add a lazy import.
- SF3D model paths are hard-coded to a sibling project. If that project moves,
  update the three constants in `sf3d_runner.py`.
- `data/work/vlm_3d/candidates/` and `data/work/vlm_3d/meshes/` are legacy
  empty directories from earlier iterations — safe to delete.
- No VLM Critic / unified scoring yet; meshes are accepted as-is from SF3D.

---

## 8. File map (relevant pieces)

```
src/patent_pipeline/
  cli.py                                 # all stages + arg parsing
  routing.py                             # patent_type routing (object / surface_pattern / ...)
  shapes.py                              # CLIP prewarm (optimize mode only)
  vlm_3d/
    loop.py                              # prepare_art3d_job + run_art3d_loop
    augmentor/
      prompt_builder.py                  # vlm_schema -> diffusion prompt
      gemini_augmentor.py                # google-genai image generation
      proxy_selector.py                  # google-genai best-candidate picker
    reconstructor/
      sf3d_runner.py                     # subprocess driver (single + batch)
    surface/loop.py                      # surface-pattern records
    legacy_optimize/                     # old CLIP-retrieval mode
scripts/
  run_sf3d_external.py                   # standalone runner in the SF3D venv
configs/
  pipeline.yaml                          # all knobs
```

That's the whole pipeline.
