import torch
try:
    from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
except ImportError:
    pass

def silhouette_mse(renders: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    # renders: [N, H, W, 4] where alpha is renders[..., 3]
    # target_masks: [N, H, W]
    alpha = renders[..., 3]
    loss = torch.nn.functional.mse_loss(alpha, target_masks)
    return loss

def combined_mesh_loss(
    mesh: "Meshes", 
    renders: torch.Tensor, 
    target_masks: torch.Tensor, 
    weights: dict
) -> dict:
    
    loss_sil = silhouette_mse(renders, target_masks)
    loss_lap = mesh_laplacian_smoothing(mesh, method="uniform")
    loss_nc = mesh_normal_consistency(mesh)
    
    total_loss = (
        weights.get('silhouette', 1.0) * loss_sil + 
        weights.get('laplacian', 0.1) * loss_lap + 
        weights.get('normal', 0.1) * loss_nc
    )
    
    return {
        "total": total_loss,
        "silhouette": loss_sil,
        "laplacian": loss_lap,
        "normal": loss_nc
    }
