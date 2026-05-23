from __future__ import annotations

from collections.abc import Iterable


def _normalize_cpc(cpc_field: object) -> list[str]:
    """Coerce a CPC field into a list of strings.

    Accepts either a single string or any iterable of objects and returns
    a list of stringified items. Used so downstream filtering logic can
    treat the CPC column uniformly regardless of upstream format.

    Args:
        cpc_field: Raw value from the ``cpc`` key of a record.

    Returns:
        List of CPC strings (possibly empty).
    """
    if isinstance(cpc_field, str):
        return [cpc_field]
    if isinstance(cpc_field, Iterable):
        return [str(x) for x in cpc_field if x is not None]
    return []


def record_matches_cpc(record: dict, cpc_prefixes: list[str]) -> bool:
    """Check whether any of a record's CPCs starts with a target prefix.

    Args:
        record: Manifest row (must contain a ``cpc`` key).
        cpc_prefixes: List of CPC prefix strings to test against. An empty
            list means "accept everything".

    Returns:
        True if no prefixes are configured or at least one CPC of the
        record starts with a configured prefix.
    """
    if not cpc_prefixes:
        return True
    cpcs = _normalize_cpc(record.get("cpc"))
    for cpc in cpcs:
        for prefix in cpc_prefixes:
            if cpc.startswith(prefix):
                return True
    return False


def has_front_view(record: dict) -> bool:
    """Return True if the record has a non-empty ``views.front`` path.

    Args:
        record: Manifest row.

    Returns:
        True when ``record['views']`` is a dict and its ``front`` entry is
        a non-empty string. False otherwise.
    """
    views = record.get("views")
    if not isinstance(views, dict):
        return False
    front = views.get("front")
    return isinstance(front, str) and len(front.strip()) > 0


def filter_records(records: list[dict], cpc_prefixes: list[str], max_samples: int) -> list[dict]:
    """Filter manifest rows to those with a front view and matching CPC.

    Args:
        records: Input manifest rows from ``data/impact_manifest.jsonl``.
        cpc_prefixes: CPC prefixes to keep (empty list keeps everything).
        max_samples: Stop after this many kept rows (0 = unlimited).

    Returns:
        A new list containing only the rows that pass both checks, capped
        at ``max_samples``.
    """
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
