# Detailed Action Log

A chronological account of every change, command, and diagnostic step taken
in this chat while bringing the Art3D pipeline to a working end-to-end state.
Items earlier than the most recent compaction are reconstructed from the
session summary; everything after is from direct tool execution.

---

## Phase 0 — Pre-compaction context (reconstructed)

Starting state: the agent was trying to install `sf3d` into the project venv
and failing. The user chose to reuse an already-working `sf3d` venv from a
sibling project at:

```
c:\Users\Sunny\OneDrive\Documents\patent-3d-viewer\external\stable-fast-3d
```

The agent then made the following changes (paraphrased from the summary):

### 0.1 External subprocess wiring
- **Created** `scripts/run_sf3d_external.py` (initial version, single-image
  only) that called `ImageTo3DFastPipeline` / `Mesh` from `sf3d.pipelines.*`.
- **Rewrote** `src/patent_pipeline/vlm_3d/reconstructor/sf3d_runner.py` to
  invoke that script as a subprocess via the external venv's `python.exe`,
  with `capture_output=True`.

### 0.2 Gemini SDK migration (google-genai)
- **Rewrote** `src/patent_pipeline/vlm_3d/augmentor/gemini_augmentor.py` to
  use the new SDK: `genai.Client(api_key=...).models.generate_content(...)`
  with `config=types.GenerateContentConfig(response_modalities=["TEXT","IMAGE"])`.
  Image bytes read from `candidate.content.parts[].inline_data.data`.
- **Rewrote** `src/patent_pipeline/vlm_3d/augmentor/proxy_selector.py` to the
  same SDK, using `gemini-2.5-flash` as the selector model.

### 0.3 Loop fallback for missing cleaned figure
- **Modified** `src/patent_pipeline/vlm_3d/loop.py` so that if
  `cleaned_figure_path` is missing/nonexistent, it falls back to
  `views.front` (honouring the user preference to never overwrite the
  original input image).

### 0.4 Legacy import isolation in `cli.py`
- **Removed** top-level imports of `parse_constraints` and the legacy
  `run_optimization` from `cli.py`.
- **Lazy-imported** `run_optimization` inside the `mode == "optimize"` branch
  of `run_shapes`.
- Fixed downstream relative-import errors in
  `src/patent_pipeline/vlm_3d/legacy_optimize/assembler/builder.py`.

### 0.5 Workers and prewarm gating
- Forced `workers = 1` for art3d mode in `run_shapes` (subprocess + CUDA was
  unsafe under threading on Windows).
- Wrapped `prewarm_clip_resources(...)` in `if mode == "optimize":` so the
  art3d path no longer triggers the pyarrow access-violation crash via
  `sentence_transformers → datasets → pyarrow`.

### 0.6 Misc
- `pip install easyocr` (needed by `masks.py`).
- Removed `sf3d` and `pytorch3d` from `requirements.txt`.

By the end of this phase the pipeline could launch without crashing but had
not yet completed a single record.

---

## Phase 1 — Post-compaction: confirm crash bypass

### 1.1 Reapply prewarm gating
- Re-confirmed the `if mode == "optimize":` wrapper around
  `prewarm_clip_resources(...)` in `cli.py::run_shapes`.

### 1.2 Smoke-test the full shapes stage
Ran (PowerShell):

```powershell
$env:PYTHONFAULTHANDLER=1
cmd /c "python -X faulthandler -u -m src.patent_pipeline.cli shapes --mode art3d > debug_out.txt 2>&1"
```

Result: `EXIT=0`. All 44 records processed without a native crash, but every
Gemini call returned:

```
404 NOT_FOUND. models/gemini-2.5-flash-image-preview is not found ...
```

### 1.3 Discover the correct image-gen model name
Ran a small `genai.Client.models.list()` script and confirmed the available
image models include `models/gemini-2.5-flash-image` (no `-preview` suffix).

### 1.4 Update the image model
- **Edited** `gemini_augmentor.py`:
  `_IMAGE_GEN_MODEL = "gemini-2.5-flash-image"` (dropped `-preview`).

---

## Phase 2 — Add `--limit` for cheap iteration

### 2.1 CLI arg + plumbing
- **Edited** `cli.py`:
  - Added `parser.add_argument("--limit", ...)`.
  - Changed `run_shapes` signature to `run_shapes(cfg, mode, limit=0)`.
  - Slice `rows = rows[:limit]` when set.
  - Pass `limit=args.limit` from `main()`.

### 2.2 Run on 1 record
```powershell
python -X faulthandler -u -m src.patent_pipeline.cli shapes --mode art3d --limit 1
```

Result: Gemini augmentation succeeded (proxies generated), but SF3D failed:

```
ModuleNotFoundError: No module named 'sf3d'
```

Root cause: the external venv has the `sf3d` package as source under the
project root, not pip-installed.

---

## Phase 3 — Make SF3D actually import

### 3.1 Inspected the external project
Found that the upstream `run.py` does:

```python
from sf3d.system import SF3D
from sf3d.utils import remove_background, resize_foreground
```

and is normally invoked from the project root.

### 3.2 Rewrite the standalone script
- **Rewrote** `scripts/run_sf3d_external.py` to mirror that import pattern
  (use `SF3D.from_pretrained` + `model.run_image([img], bake_resolution=1024,
  remesh="none", vertex_count=-1)`), and to set `--input_image` /
  `--output_mesh` / `--device` args.

### 3.3 Make the import resolve
- **Edited** `sf3d_runner.py` to:
  - Add `SF3D_PROJECT_ROOT` constant.
  - Set `cwd=SF3D_PROJECT_ROOT` on the subprocess.
  - Prepend `SF3D_PROJECT_ROOT` to `PYTHONPATH` in the child env (cwd alone
    doesn't put the dir on `sys.path`).

### 3.4 Robust proxy-selector parsing
- **Edited** `proxy_selector.py` to fall back to the first integer found in
  the response when Gemini prepends prose to the digit. Catches any error and
  defaults to candidate 0.

Status at end of Phase 3: ready to re-run, but the user's machine then froze
during that run.

---

## Phase 4 — Diagnose the freeze and de-risk the run

### 4.1 Diagnosis
Identified four contributors:

1. **Two GPU consumers at once**: the main `.venv` had a torch/CUDA context
   alongside the SF3D subprocess. On Windows that thrashes the WDDM driver.
2. **Cold model load per record**: SF3D + rembg would reload for all 44 jobs,
   one subprocess each.
3. **Buffered subprocess output**: `capture_output=True` held everything in
   RAM.
4. **rembg first-run download** of `u2net` weights during the very first job.

### 4.2 Strategy
Two-phase Art3D run:
- **Phase A**: Gemini-only per record, no GPU touched in the main process.
- **Phase B**: a *single* SF3D subprocess that loads the model once and
  processes all jobs in-process, with streamed stdout.

### 4.3 Refactor `vlm_3d/loop.py`
- **Edited** to split the loop:
  - `prepare_art3d_job(record, config, work_dir)` returns
    `(updated_record, sf3d_job_or_None)` and does no CUDA work.
  - `run_art3d_loop(...)` kept as a thin wrapper for legacy callers; it now
    delegates to `prepare_art3d_job` + a single-image
    `run_sf3d_reconstruction`.

### 4.4 Refactor `vlm_3d/reconstructor/sf3d_runner.py`
- **Rewrote** to add:
  - `_build_env()` — common `PYTHONPATH` + `PYTHONUNBUFFERED` setup.
  - `_stream_subprocess(cmd)` — `Popen` + line-buffered stdout streamed to
    the parent in real time, returns `(rc, full_stdout)`.
  - `run_sf3d_reconstruction(...)` — single-image, uses
    `_stream_subprocess`.
  - `run_sf3d_batch(jobs, device)` — writes a temp JSON manifest of
    `{input, output}` records, calls the runner with `--jobs <path>`, parses
    the trailing `[sf3d-ext] RESULTS=<json>` line to merge per-job
    `ok` / `error` / `output` back into the caller's job dicts.

### 4.5 Rewrite the standalone runner script
- **Rewrote** `scripts/run_sf3d_external.py` to support **two modes**:
  - `--input_image / --output_mesh` (single, unchanged).
  - `--jobs JOBS.json` where the script loads SF3D **once**, iterates over
    all jobs, calls `torch.cuda.empty_cache()` between them, prints
    `[sf3d-ext] RESULTS=<json>` at the end.
  - Per-job errors are caught and recorded; the script returns 0 unless the
    manifest is missing/empty.

### 4.6 Two-phase shapes orchestration in `cli.py`
- **Edited** `run_shapes` to dispatch to
  `_run_shapes_art3d_batched(cfg, rows, work_dir, stage)` when
  `mode == "art3d"`.
- **Added** `_run_shapes_art3d_batched` which:
  - Sets `os.environ["CUDA_VISIBLE_DEVICES"] = ""` in the parent process so
    the main Python cannot allocate a CUDA context.
  - **Phase A**: loops `rows`, routes via `route_patent(...)`. Surface-pattern
    records go through `run_surface_pattern_loop`; everything else goes
    through `prepare_art3d_job(...)`. Failures are caught and tagged as
    `art3d_result = "prepare_failed: ..."`.
  - **Phase B**: if any jobs were produced, calls
    `run_sf3d_batch(jobs, device="cuda")` once. Results are merged back into
    each record by `patent_id`, populating `mesh_path` and
    `art3d_result = "processed"`, or `"sf3d_failed: ..."` on failure.
  - Writes `shaped_manifest.jsonl` plus a `failures_shapes.jsonl` if any
    rows failed.
- Kept the `optimize` branch unchanged.

### 4.7 Validation
Ran:

```powershell
$env:PYTHONFAULTHANDLER=1
cmd /c "python -X faulthandler -u -m src.patent_pipeline.cli shapes --mode art3d --limit 1 > debug_out.txt 2>&1"
```

Result: `EXIT=0`. Gemini produced proxies, selector picked
`proxy_1.png`, batched SF3D subprocess loaded the model once and wrote
`data/work/vlm_3d/reconstructed_meshes/D0937859.glb`. First end-to-end
success, no freeze.

---

## Phase 5 — Confirm output locations

- Listed `data/work/vlm_3d/` and verified artifacts are inside the workspace:
  - `data/work/vlm_3d/proxies/D0937859/` (Gemini outputs)
  - `data/work/vlm_3d/reconstructed_meshes/D0937859.glb` (~874 KB)
  - `data/work/shaped_manifest.jsonl` (row with `mesh_path`, …)
- Noted two leftover empty dirs from earlier iterations:
  `data/work/vlm_3d/candidates/` and `data/work/vlm_3d/meshes/` (safe to
  delete; not touched).

---

## Phase 6 — Add `--patent-id` and re-run on a different patent

### 6.1 CLI arg
- **Edited** `cli.py`:
  - Added `parser.add_argument("--patent-id", default=None, ...)`.
  - Extended `run_shapes` signature to
    `run_shapes(cfg, mode, limit=0, patent_id=None)`.
  - Filters `rows` by `patent_id` (comma-separated allowed) before applying
    `limit`.
  - `main()` passes `patent_id=args.patent_id`.

### 6.2 Pick a record and run
- Listed posed manifest IDs (44 total) and picked `D0908314`.
- Ran:
  ```powershell
  python -X faulthandler -u -m src.patent_pipeline.cli shapes --mode art3d --patent-id D0908314
  ```
- Result: `EXIT=0`. Gemini produced 3 proxies, selector picked
  `proxy_2.png`, batched SF3D produced
  `data/work/vlm_3d/reconstructed_meshes/D0908314.glb`.

---

## Phase 7 — Documentation

- **Created** `WALKTHROUGH.md` summarising the architecture, all eight key
  design decisions, run instructions, output layout, and a file map.
- **Created** `DETAILED.md` (this file) — chronological action log.

---

## Appendix A — Files created or modified

| Path                                                                       | Action                                  |
|----------------------------------------------------------------------------|-----------------------------------------|
| `scripts/run_sf3d_external.py`                                             | Created → rewritten (dual single/batch) |
| `src/patent_pipeline/vlm_3d/reconstructor/sf3d_runner.py`                  | Rewritten (batch + streaming)           |
| `src/patent_pipeline/vlm_3d/augmentor/gemini_augmentor.py`                 | Rewritten (new SDK, correct model name) |
| `src/patent_pipeline/vlm_3d/augmentor/proxy_selector.py`                   | Rewritten (new SDK, robust parsing)     |
| `src/patent_pipeline/vlm_3d/loop.py`                                       | Refactored (prepare_art3d_job + loop)   |
| `src/patent_pipeline/vlm_3d/legacy_optimize/assembler/builder.py`          | Relative-import fixes                   |
| `src/patent_pipeline/cli.py`                                               | Two-phase art3d, `--limit`, `--patent-id`, lazy legacy imports, prewarm gating |
| `requirements.txt`                                                         | Removed `sf3d` and `pytorch3d`          |
| `WALKTHROUGH.md`                                                           | Created                                 |
| `DETAILED.md`                                                              | Created (this file)                     |

## Appendix B — Diagnostic / one-off commands run

- `pip install easyocr`
- `python -c "from google import genai; ... .models.list()"` to discover
  `gemini-2.5-flash-image`.
- `python -c "...json.loads..."` to list `patent_id` values in the posed
  manifest.
- Multiple iterations of:
  ```
  $env:PYTHONFAULTHANDLER=1
  cmd /c "python -X faulthandler -u -m src.patent_pipeline.cli shapes --mode art3d [...] > debug_out.txt 2>&1"
  Get-Content debug_out.txt -Tail N
  ```
  This pattern was used because PowerShell + tqdm buffering was hiding the
  Windows native crash; `cmd /c "... > file 2>&1"` plus `faulthandler` made
  the access violation visible.
- `Get-ChildItem` + `Get-Item` to confirm output artifact paths and sizes.

## Appendix C — Mistakes and recoveries

1. **Wrong Gemini model name** (`gemini-2.5-flash-image-preview`) → fixed by
   listing models and switching to `gemini-2.5-flash-image`.
2. **Wrong SF3D import paths** (`sf3d.pipelines.image_to_3d_pipeline.*`) → the
   external project exposes `sf3d.system.SF3D` instead; runner rewritten.
3. **`ModuleNotFoundError: sf3d`** in the subprocess → fixed by setting
   `cwd=SF3D_PROJECT_ROOT` *and* prepending it to `PYTHONPATH`.
4. **System freeze** during full-manifest runs → fixed by GPU isolation
   (`CUDA_VISIBLE_DEVICES=""` in parent), single batched SF3D subprocess,
   streamed stdout, and `empty_cache()` between jobs.
5. **pyarrow native access violation** → bypassed by skipping
   `prewarm_clip_resources(...)` outside `--mode optimize`.
6. **Two prior turns where edits rendered as code fences but tool calls did
   not actually fire** → noticed by re-reading the files; redone with
   explicit `multi_replace_string_in_file` / `create_file` calls.

End of log.
