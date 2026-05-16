def rerank_candidates(candidates: list[dict], alpha: float = 0.5) -> int:
    # candidates is a list of dicts: {"numeric_loss": float, "critic_score": float, "mesh": ...}
    best_idx = 0
    best_score = -float('inf')
    
    for i, cand in enumerate(candidates):
        # We want to minimize loss, maximize critic_score.
        # Combined score = critic_score - alpha * numeric_loss
        score = cand["critic_score"] - alpha * cand["numeric_loss"]
        if score > best_score:
            best_score = score
            best_idx = i
            
    return best_idx
