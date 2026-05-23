def rerank_candidates(candidates: list[dict], alpha: float = 0.5) -> int:
    """Pick the best candidate by combining the critic score with a loss penalty.

    Each candidate's combined score is ``critic_score - alpha * numeric_loss``;
    the candidate with the highest combined score wins.

    Args:
        candidates: List of dicts containing ``numeric_loss`` and
            ``critic_score`` floats.
        alpha: Weight applied to the numeric loss penalty.

    Returns:
        Index into ``candidates`` of the best entry (defaults to 0 when
        the list is empty since ``best_idx`` is pre-initialized).
    """
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
