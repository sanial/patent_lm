# Patent 2D → 3D Supervision Pipeline

Convert **US design-patent line-art figures** (from the IMPACT dataset) into
**textured 3D meshes** (`.glb`) for use as weak 3D supervision. The current
pipeline produces, for each patent, a photoreal "proxy" image (via Gemini) and a
reconstructed mesh (via Stable-Fast-3D), packaged into a single JSONL manifest.

---

## TL;DR

```powershell
.\.venv\Scripts\Activate.ps1
$env:GEMINI_API_KEY = "..."   # required for the art3d mode

python -m src.patent_pipeline.cli shapes --mode art3d --patent-id D0915081
# → data/work/vlm_3d/proxies/D0915081/proxy_*.png
# → data/work/vlm_3d/reconstructed_meshes/D0915081.glb
```

---

## What it does

For each patent in the IMPACT subset:

1. **ingest / filter / masks / poses** — build a metadata manifest, drop bad
   rows, generate per-instance masks, and produce a VLM-derived structural
   schema (parts, relations, symmetries).
2. **shapes (`--mode art3d`)** — generate photoreal proxy images and a 3D mesh:
   - **Phase A (per record, CPU only):** `gemini-2.5-flash-image` turns the
     cleaned patent figure + schema-derived prompt into N=3 candidate `.png`
     proxies. `gemini-2.5-flash` then picks the best one.
   - **Phase B (batched, GPU):** a single subprocess launched in an external
     `stable-fast-3d` virtualenv loads SF3D once and reconstructs all queued
     meshes, streaming results back as `.glb` files.
3. **package** — writes the final
   [data/output/patent_3d_supervision.jsonl](data/output/patent_3d_supervision.jsonl).

The two-phase split (with `CUDA_VISIBLE_DEVICES=""` on the parent process)
exists to avoid a Windows CUDA hang that happens when two processes share a GPU
context.

```
TIF line-art ──► Gemini image-gen ──► proxy_{0..2}.png ──► selector
                                                              │
                                              best proxy ◄────┘
                                                  │
                                                  ▼
                                          SF3D subprocess (batched)
                                                  │
                                                  ▼
                                          reconstructed_meshes/*.glb
```

---

## Why Gemini and not ControlNet?

The augmentor's role — taking patent line-art as structural conditioning and
producing a photorealistic image that preserves its contours — is filled by
`gemini-2.5-flash-image` receiving the figure as a multimodal input alongside a
text prompt. There is **no real ControlNet** in this pipeline; any docstring
mentioning it is a historical note.

---

## How proxy images are made

SF3D was trained on photorealistic product photos, not patent line-art, so a
bridge step turns the drawing into something SF3D can reconstruct:

1. The upstream `poses` stage emits a structured `vlm_schema`
   (`{category, parts, materials, view, ...}`).
   [prompt_builder.py](src/patent_pipeline/vlm_3d/augmentor/prompt_builder.py)
   renders that to a short natural-language description.
2. [gemini_augmentor.py](src/patent_pipeline/vlm_3d/augmentor/gemini_augmentor.py)
   sends `[instruction, line_art_PIL_image]` to `gemini-2.5-flash-image` with
   `response_modalities=["TEXT", "IMAGE"]`, N times (default N=3). Each call
   yields one PIL image, decoded from the `inline_data` part of the response
   and saved as `proxy_<i>.png` via `PIL.Image.save`.
3. [proxy_selector.py](src/patent_pipeline/vlm_3d/augmentor/proxy_selector.py)
   makes a second call to `gemini-2.5-flash` (text-only) passing the line art
   and all N candidates, asking for the index of the best one. That path
   becomes `best_proxy_path`.
4. Only `best_proxy_path` is fed to SF3D in Phase B; the other candidates are
   retained so the selector's choice is auditable.

---

## Dataset layout

| Path | Contents |
|---|---|
| [data/external/IMPACT/](data/external/IMPACT) | Raw IMPACT figures (TIFs grouped by patent) |
| [data/impact_manifest.jsonl](data/impact_manifest.jsonl) | One row per patent after ingest |
| [data/work/filtered_manifest.jsonl](data/work/filtered_manifest.jsonl) | After `filter` |
| [data/work/masked_manifest.jsonl](data/work/masked_manifest.jsonl) | + `masks_path` per row |
| [data/work/posed_manifest.jsonl](data/work/posed_manifest.jsonl) | + `vlm_schema` per row |
| [data/work/shaped_manifest.jsonl](data/work/shaped_manifest.jsonl) | + proxy paths + `mesh_path` |
| [data/work/vlm_3d/proxies/](data/work/vlm_3d/proxies/) | Gemini-generated `.png` proxies per patent |
| [data/work/vlm_3d/reconstructed_meshes/](data/work/vlm_3d/reconstructed_meshes/) | SF3D `.glb` outputs |
| [data/output/patent_3d_supervision.jsonl](data/output/patent_3d_supervision.jsonl) | Final packaged supervision |

---

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your Gemini key (required for `shapes --mode art3d`):

```powershell
$env:GEMINI_API_KEY = "..."
```

### External SF3D venv (required for art3d)

The `art3d` mode does **not** install SF3D into this project's venv. Instead it
shells out to an external venv at:

```
c:\Users\Sunny\OneDrive\Documents\patent-3d-viewer\external\stable-fast-3d\.venv
```

Paths are hardcoded in
[src/patent_pipeline/vlm_3d/reconstructor/sf3d_runner.py](src/patent_pipeline/vlm_3d/reconstructor/sf3d_runner.py).
Update them if your sf3d install lives elsewhere. The runner script that
executes inside that venv is
[scripts/run_sf3d_external.py](scripts/run_sf3d_external.py).

---

## Run

### Full chain

```powershell
python -m src.patent_pipeline.cli ingest
python -m src.patent_pipeline.cli filter
python -m src.patent_pipeline.cli masks
python -m src.patent_pipeline.cli poses
python -m src.patent_pipeline.cli shapes --mode art3d
python -m src.patent_pipeline.cli package
```

### Selective re-runs

```powershell
# one patent
python -m src.patent_pipeline.cli shapes --mode art3d --patent-id D0908314

# multiple
python -m src.patent_pipeline.cli shapes --mode art3d --patent-id D0908314,D0915081

# first N rows
python -m src.patent_pipeline.cli shapes --mode art3d --limit 5
```

### Recommended diagnostic invocation (Windows)

PowerShell hides some native stdout. Use this to capture everything:

```powershell
$env:PYTHONFAULTHANDLER=1
cmd /c "python -X faulthandler -u -m src.patent_pipeline.cli shapes --mode art3d --patent-id D0915081 > debug_out.txt 2>&1"
Get-Content debug_out.txt -Tail 30
```

### Legacy `optimize` mode

The original primitive-fit + CLIP/ShapeNet retrieval path is still available via
`--mode optimize`. It uses `data/shapenet_proxy/` and is independent of Gemini /
SF3D.

```powershell
python -m src.patent_pipeline.cli shapes --mode optimize
```

---

## Output schema (final JSONL)

```jsonc
{
  "patent_id": "D0908314",
  "title": "...",
  "caption": "...",
  "claim": "...",
  "date": "20210126",
  "cpc": ["..."],
  "views": { "front": "...TIF", "back": "...TIF", ... },
  "masks_path": "data/work/masks/D0908314_masks.npz",
  "vlm_schema": { "parts": [...], "relations": [...], "symmetries": [...] },
  "proxy_image_paths": ["data/work/vlm_3d/proxies/D0908314/proxy_0.png", ...],
  "best_proxy_path":   "data/work/vlm_3d/proxies/D0908314/proxy_2.png",
  "mesh_path":         "data/work/vlm_3d/reconstructed_meshes/D0908314.glb",
  "art3d_result": "processed"
}
```

---

## Patents reconstructed so far

| Patent | Selected proxy | Mesh |
|---|---|---|
| D0937859 | proxy_1 | [D0937859.glb](data/work/vlm_3d/reconstructed_meshes/D0937859.glb) |
| D0908314 | proxy_2 | [D0908314.glb](data/work/vlm_3d/reconstructed_meshes/D0908314.glb) |
| D0915081 | proxy_0 | [D0915081.glb](data/work/vlm_3d/reconstructed_meshes/D0915081.glb) |
| D0913312 | proxy_1 | [D0913312.glb](data/work/vlm_3d/reconstructed_meshes/D0913312.glb) |
| D0930094 | proxy_0 | [D0930094.glb](data/work/vlm_3d/reconstructed_meshes/D0930094.glb) |

---

## Configuration

All knobs live in [configs/pipeline.yaml](configs/pipeline.yaml). Notable ones:

- `vlm_3d.model_name` — selector model (default `gemini-2.5-flash`)
- `vlm_3d.num_candidates` — proxy count per patent (default `3`)
- `masks.workers` / `pose.workers` / `shape.workers` — parallelism for the
  non-GPU stages (art3d forces 1 worker internally)

---

## Known follow-ups

- The VLM **critic** / unified scoring stages are stubs.
- Full 44-record art3d run with the new batched architecture has not yet been
  executed end-to-end; five patents are individually validated.
- Empty legacy dirs `data/work/vlm_3d/candidates/` and `data/work/vlm_3d/meshes/`
  can be removed.

---

## Repo map

```
configs/pipeline.yaml
scripts/
  run_sf3d_external.py        # executes inside external sf3d venv
src/patent_pipeline/
  cli.py                      # stage dispatcher; orchestrates Phase A/B
  ingest.py / filtering.py / masks.py / pose.py / shapes.py
  packaging.py
  shape_index.py              # ShapeNet proxy index (optimize mode)
  vlm_3d/
    loop.py                   # prepare_art3d_job, run_art3d_loop
    augmentor/
      gemini_augmentor.py     # Gemini image generation
      proxy_selector.py       # Gemini "best proxy" picker
    reconstructor/
      sf3d_runner.py          # parent-side SF3D subprocess driver
    parser/ critic/ renderer/ reranker/ assembler/
data/
  external/IMPACT/            # raw figures
  work/                       # intermediate manifests + proxies + meshes
  output/patent_3d_supervision.jsonl
```
