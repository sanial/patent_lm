from __future__ import annotations

from collections.abc import Iterable


def _normalize_cpc(cpc_field: object) -> list[str]:
    if isinstance(cpc_field, str):
        return [cpc_field]
    if isinstance(cpc_field, Iterable):
        return [str(x) for x in cpc_field if x is not None]
    return []


def record_matches_cpc(record: dict, cpc_prefixes: list[str]) -> bool:
    if not cpc_prefixes:
        return True
    cpcs = _normalize_cpc(record.get("cpc"))
    for cpc in cpcs:
        for prefix in cpc_prefixes:
            if cpc.startswith(prefix):
                return True
    return False


def has_front_view(record: dict) -> bool:
    views = record.get("views")
    if not isinstance(views, dict):
        return False
    front = views.get("front")
    return isinstance(front, str) and len(front.strip()) > 0


def filter_records(records: list[dict], cpc_prefixes: list[str], max_samples: int) -> list[dict]:
    kept: list[dict] = []
    for row in records:
        if not has_front_view(row):
            continue
        if not record_matches_cpc(row, cpc_prefixes):
            continue
        kept.append(row)
        if max_samples > 0 and len(kept) >= max_samples:
            break
    return kept
