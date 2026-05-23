import torch
from pathlib import Path
import cv2
import numpy as np

try:
    from pytorch3d.io import IO
except ImportError:
    pass

from .assembler.builder import build_assembly
from .renderer.cameras import get_silhouette_renderer
from .renderer.losses import combined_mesh_loss
from .critic.client import critique_render
from .reranker.scorer import rerank_candidates
from .parser.schema import ConstraintSchema

def _save_renders(renders: torch.Tensor, out_dir: Path, prefix: str) -> list[Path]:
    """Save a batch of rendered views to disk as PNGs.

    Args:
        renders: ``[N, H, W, 4]`` float tensor in [0, 1].
        out_dir: Output directory (created on demand).
        prefix: Filename prefix; each render is saved as
            ``<prefix>_view_<i>.png``.

    Returns:
        List of saved file paths in render order.
    """
    paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    # renders: [N, H, W, 4]
    for i in range(renders.shape[0]):
        img = renders[i].detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        p = out_dir / f"{prefix}_view_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths

def run_optimization(
    record: dict,
    schema: ConstraintSchema,
    config: "Vlm3dConfig",
    work_dir: Path
) -> dict:
    """Legacy differentiable-render optimization loop for one record.

    Builds an initial primitive assembly from the parsed schema, renders it
    with a silhouette renderer across four canonical cameras, scores each
    candidate with both a numeric loss and a VLM critic, and saves the
    best mesh as an ``.obj`` file.

    Args:
        record: Manifest row with ``patent_id`` and ``masks_path``.
        schema: VLM-parsed constraint schema describing the parts.
        config: VLM-3D configuration (provides ``num_candidates``,
            ``model_name``, ``loss_weights``).
        work_dir: Working directory for renders and final mesh outputs.

    Returns:
        A shallow copy of ``record`` augmented with ``vlm_constraints``,
        ``best_candidate_idx``, ``critic_report``, and ``mesh_path``.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Build initial mesh
    initial_mesh = build_assembly(schema, device)
    
    # 2. Get masks (target)
    masks_path = Path(record.get("masks_path", ""))
    if masks_path.exists():
        target_masks = torch.tensor(np.load(masks_path)["masks"], dtype=torch.float32, device=device)
        # Ensure it matches camera views (very simplified here, just mapping mask 0 to view 0 for prototype)
        # Real implementation would need mask alignment
        target_masks = target_masks[:4] # Take up to 4 masks
        # Resize to square 256x256 for renderer compatibility
        target_masks = torch.nn.functional.interpolate(target_masks.unsqueeze(1), size=(256, 256), mode="bilinear", align_corners=False).squeeze(1)
        if target_masks.shape[0] < 4:
            # pad
            pad = torch.zeros((4 - target_masks.shape[0], *target_masks.shape[1:]), device=device)
            target_masks = torch.cat([target_masks, pad], dim=0)
    else:
        target_masks = torch.zeros((4, 256, 256), device=device)
        
    renderer = get_silhouette_renderer(device, image_size=target_masks.shape[-1])
    
    candidates = []
    
    # Generate multiple candidates
    for cand_idx in range(config.num_candidates):
        # clone mesh
        mesh = initial_mesh.clone()
        # extend to batch size 4 for the 4 cameras
        render_mesh = mesh.clone().extend(4)
        
        # Optimization loop (mocked for simplicity here, just doing one forward pass)
        # Real code would use torch.optim.Adam(mesh.verts_list())
        renders = renderer(render_mesh)
        
        loss_dict = combined_mesh_loss(mesh, renders, target_masks, config.loss_weights)
        
        cand_dir = work_dir / "candidates" / f"{record['patent_id']}_{cand_idx}"
        view_paths = _save_renders(renders, cand_dir, "render")
        
        critic_schema = critique_render(
            rendered_view_paths=view_paths, 
            constraints=schema, 
            model_name=config.model_name
        )
        
        candidates.append({
            "numeric_loss": loss_dict["total"].item(),
            "critic_score": critic_schema.rerank_score,
            "critic_schema": critic_schema.model_dump(),
            "mesh": mesh
        })
        
    best_idx = rerank_candidates(candidates)
    best_cand = candidates[best_idx]
    
    # Save best mesh
    mesh_out_path = work_dir / "meshes" / f"{record['patent_id']}.obj"
    mesh_out_path.parent.mkdir(parents=True, exist_ok=True)
    io = IO()
    io.save_mesh(best_cand["mesh"], str(mesh_out_path))
    
    updated = dict(record)
    updated["vlm_constraints"] = schema.model_dump()
    updated["best_candidate_idx"] = best_idx
    updated["critic_report"] = best_cand["critic_schema"]
    updated["mesh_path"] = str(mesh_out_path)
    
    return updated
