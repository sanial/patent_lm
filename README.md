# Patent 2D->3D Supervision Pipeline (IMPACT-first)

This repo gives you a practical starter pipeline for the exact dataset recipe you outlined:

1. Filter IMPACT metadata to a focused CPC slice (for example `A47` furniture)
2. Generate instance masks from patent figure line-art
3. Derive weak 3D pose labels (bbox lifting, with optional COLMAP hook)
4. Derive weak 3D shape labels (primitive default, CLIP retrieval hook)
5. Package outputs as JSONL and optional Hugging Face dataset

## Bootstrap + IMPACT repo clone

```powershell
$env:PYTHONPATH = "src"
python -m patent_pipeline.cli bootstrap
python -m patent_pipeline.cli download-impact --impact-dir data/external
python -m patent_pipeline.cli ingest-impact --max-samples 500
python -m patent_pipeline.cli build-proxy-shape-index --config configs/pipeline.yaml
```

This creates:

- `configs/pipeline.yaml` (if missing)
- `data/impact_manifest.jsonl` sample template (if missing)
- `data/external/IMPACT` cloned repository
- `data/impact_manifest.jsonl` generated from IMPACT metadata
- `data/shapenet_proxy/shapenet_embeddings.npy` and `data/shapenet_proxy/shapenet_mesh_index.json`

## Parallel workers

You can speed up large runs by increasing these values in `configs/pipeline.yaml`:

- `masks.workers`
- `pose.workers`
- `shape.workers`

## Expected input manifest

Create `data/impact_manifest.jsonl` with one JSON object per line.

Example row:

```json
{
  "patent_id": "D0912345",
  "cpc": ["A47C3/00"],
  "caption": "A chair with curved backrest.",
  "views": {
    "front": "data/images/D0912345/front.png",
    "side": "data/images/D0912345/side.png",
    "top": "data/images/D0912345/top.png",
    "perspective": "data/images/D0912345/perspective.png"
  }
}
```

Only `patent_id` and `views.front` are mandatory for the starter pipeline.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure

Edit config:

```powershell
notepad configs/pipeline.yaml
```

## Run end-to-end (500 sample pilot)

```powershell
$env:PYTHONPATH = "src"
python -m patent_pipeline.cli all --config configs/pipeline.yaml
```

## Run stages separately

```powershell
$env:PYTHONPATH = "src"
python -m patent_pipeline.cli filter --config configs/pipeline.yaml
python -m patent_pipeline.cli masks --config configs/pipeline.yaml
python -m patent_pipeline.cli poses --config configs/pipeline.yaml
python -m patent_pipeline.cli shapes --config configs/pipeline.yaml
python -m patent_pipeline.cli package --config configs/pipeline.yaml
```

## Output schema

Each final row includes:

- `image`: front view path
- `masks_path`: path to `.npz` instance masks
- `poses`: list of weak 3D pose labels per instance
- `shapes`: list of weak 3D shape labels per instance
- `patent_id`, `caption`, `cpc`

## Notes

- Default setup is optimized for a proof-of-concept dataset build.
- If `COLMAP` is available and enabled, a hook runs it per sample with multi-view images.
- `clip_retrieval` mode performs real nearest-neighbor retrieval with `sentence-transformers` using masked object crops.
- Long-running stages write per-stage failure logs in `data/work/failures_*.jsonl` and continue processing remaining samples.
