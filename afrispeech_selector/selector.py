"""Select languages from the catalog.

Two entry points:

* :func:`filter_catalog` — apply hours / country / split constraints.
* :func:`select_top` — rank by hours and pick the top *N* with optional
  country proportionality ("one strong language per country first, then fill").

Selection is pure (no I/O): it operates on :class:`~afrispeech_selector.catalog.LanguageEntry`
objects and returns a list of them. Pulling actual audio happens in
:mod:`afrispeech_selector.builder`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .catalog import LanguageEntry, load_catalog, resolve_dataset


def filter_catalog(
    entries: Iterable[LanguageEntry] | None = None,
    *,
    min_hours: float = 0.0,
    max_hours: float | None = None,
    min_clips: int = 0,
    countries: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    split: str = "train",
    require_split: bool = True,
) -> list[LanguageEntry]:
    """Drop entries that don't meet the given criteria.

    Args:
        min_hours / max_hours: keep languages whose total hours fall in range.
        min_clips: keep languages with at least this many clips.
        countries: if given, keep only these ISO 3166 country codes.
        datasets: if given, keep only these source datasets (keys like ``waxal``
            or full HF ids). Unknown keys are ignored.
        split: which split the samples will be drawn from later.
        require_split: drop languages that have zero clips in ``split``
            (e.g. a language with an empty test set is useless if split="test").
    """
    entries = list(entries) if entries is not None else load_catalog()
    country_set = {c.upper() for c in countries} if countries else None
    if datasets:
        wanted = set()
        for d in datasets:
            try:
                wanted.add(resolve_dataset(d))
            except Exception:
                continue
        entries = [e for e in entries if e.dataset_id in wanted]
        if not entries:
            return []

    out = []
    for e in entries:
        if e.hours < min_hours:
            continue
        if max_hours is not None and e.hours > max_hours:
            continue
        if e.clips < min_clips:
            continue
        if country_set is not None and e.country not in country_set:
            continue
        if require_split and split != "all" and e.split_size(split) <= 0:
            continue
        out.append(e)
    return out


def select_top(
    entries: Iterable[LanguageEntry],
    n: int,
    *,
    proportional: bool = True,
    max_per_country: int | None = None,
) -> list[LanguageEntry]:
    """Pick the top ``n`` languages, ranked by hours.

    When ``proportional`` is True we round-robin across countries: every country
    contributes its strongest language before any country contributes a second
    one, maximising geographic diversity. ``max_per_country`` caps how many a
    single country may contribute (e.g. 2). Within each round, countries are
    visited in order of their strongest remaining language so stronger countries
    still get priority.

    When ``proportional`` is False this is a plain top-N by hours.
    """
    ranked = sorted(entries, key=lambda e: e.hours, reverse=True)
    if n <= 0:
        return []
    if not proportional:
        if max_per_country is None:
            return ranked[:n]
        # still honour the per-country cap on a pure ranking
        counts: dict[str, int] = defaultdict(int)
        out = []
        for e in ranked:
            if counts[e.country] >= max_per_country:
                continue
            out.append(e)
            counts[e.country] += 1
            if len(out) >= n:
                break
        return out

    # Group by country, each group sorted strongest-first.
    by_country: dict[str, list[LanguageEntry]] = defaultdict(list)
    for e in ranked:
        by_country[e.country].append(e)

    # Visit countries in order of their strongest language.
    country_order = sorted(
        by_country, key=lambda c: by_country[c][0].hours, reverse=True
    )

    selected: list[LanguageEntry] = []
    cursor: dict[str, int] = defaultdict(int)
    cap = max_per_country if max_per_country is not None else math.inf

    progressed = True
    while len(selected) < n and progressed:
        progressed = False
        depth_floor = min(cursor.values()) if cursor else 0
        for country in country_order:
            langs = by_country[country]
            idx = cursor[country]
            # round-robin: only advance a country that is at the current depth
            if idx != depth_floor:
                continue
            if idx >= len(langs) or idx >= cap:
                continue
            selected.append(langs[idx])
            cursor[country] += 1
            progressed = True
            if len(selected) >= n:
                break
    return selected


def plan_samples(
    entries: Iterable[LanguageEntry],
    *,
    per_language: int | None,
    max_seconds: float | None = None,
    min_clip_seconds: float | None = None,
    max_clip_seconds: float | None = None,
    split: str = "train",
) -> list[dict]:
    """Estimate how many clips / how much audio will be pulled per language.

    Caps applied (whichever binds first): ``per_language`` clip count and
    ``max_seconds`` duration budget. A per-clip length window
    (``min_clip_seconds``/``max_clip_seconds``) is treated as a precondition:
    the effective average clip length used for the hours↔clips conversion is
    clamped into the window, so the count needed to reach an hour target shifts
    accordingly. Estimates use the average clip length (hours/clips) — so for a
    length window or duration budget they are approximate; the actual build
    accumulates real per-clip durations until closest to target. One dict/lang.
    """
    lo = min_clip_seconds if min_clip_seconds is not None else 0.0
    hi = max_clip_seconds if max_clip_seconds is not None else float("inf")
    plan = []
    for e in entries:
        available = e.split_size(split)
        avg = (e.hours * 3600 / e.clips) if e.clips else 0.0
        # Clamp the assumed clip length into the requested window.
        eff = min(max(avg, lo), hi) if avg > 0 else 0.0
        avail_seconds = available * eff

        planned = available
        if per_language is not None:
            planned = min(planned, per_language)
        if max_seconds is not None and eff > 0:
            planned = min(planned, max(1, round(max_seconds / eff)))

        planned_seconds = round(planned * eff, 1)
        if max_seconds is not None:
            planned_seconds = round(min(planned_seconds, avail_seconds), 1)

        row = e.as_dict()
        row.update(split=split, available=available, planned=planned,
                   planned_seconds=planned_seconds,
                   planned_hours=round(planned_seconds / 3600, 2))
        plan.append(row)
    return plan
