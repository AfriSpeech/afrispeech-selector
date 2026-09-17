"""Tests for the pure selection logic (no network / datasets dependency)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from afrispeech_selector.catalog import load_catalog
from afrispeech_selector.selector import filter_catalog, plan_samples, select_top


def test_catalog_loads():
    cat = load_catalog()
    assert len(cat) == 369
    twi = next(e for e in cat if e.subset == "twi_twi")
    assert twi.country == "GH" and abs(twi.hours - 50.32) < 1e-6
    assert twi.train + twi.val + twi.test <= twi.clips
    # multi-source catalog: one entry per dataset carries its source id
    from collections import Counter
    assert Counter(e.dataset for e in cat) == {
        "afrispeech": 142, "waxal": 34, "omnilingual": 174, "open-bible": 19}
    waxal = next(e for e in cat if e.dataset == "waxal")
    assert waxal.dataset_id == "google/WaxalNLP"
    assert waxal.dataset_name == "Google WaxalNLP"
    assert next(e for e in cat if e.dataset == "open-bible").dataset != "afrispeech"


def test_min_max_hours_filter():
    pool = filter_catalog(min_hours=20, max_hours=40)
    assert all(20 <= e.hours <= 40 for e in pool)
    assert all(e.hours >= 20 for e in pool)
    # malagasy (61h) excluded by max; luvale (0.07h) excluded by min
    subsets = {e.subset for e in pool}
    assert "malagasy_mlg" not in subsets
    assert "luvale_lue" not in subsets


def test_country_filter_and_split_requirement():
    pool = filter_catalog(countries=["gh"], split="train")
    assert pool and all(e.country == "GH" for e in pool)
    # split="test" should drop languages with an empty test set
    no_test = [e for e in load_catalog() if e.test == 0]
    assert no_test  # sanity
    pool_test = filter_catalog(split="test")
    assert all(e.test > 0 for e in pool_test)


def test_pure_top_n_by_hours():
    cat = load_catalog()
    top3 = select_top(cat, 3, proportional=False)
    assert [e.subset for e in top3] == [
        e.subset for e in sorted(cat, key=lambda x: x.hours, reverse=True)[:3]
    ]


def test_proportional_one_per_country_first():
    cat = load_catalog()
    chosen = select_top(cat, 10, proportional=True)
    # First pass must be 10 distinct countries (one per country before seconds).
    countries = [e.country for e in chosen]
    assert len(set(countries)) == len(countries) == 10


def test_max_per_country_cap():
    cat = load_catalog()
    chosen = select_top(cat, 30, proportional=True, max_per_country=2)
    from collections import Counter
    counts = Counter(e.country for e in chosen)
    assert all(v <= 2 for v in counts.values())


def test_proportional_strongest_country_first():
    cat = load_catalog()
    chosen = select_top(cat, 5, proportional=True)
    # The very first pick is the single strongest language overall.
    strongest = max(cat, key=lambda e: e.hours)
    assert chosen[0].subset == strongest.subset


def test_plan_samples_duration_budget():
    cat = load_catalog()
    twi = next(e for e in cat if e.subset == "twi_twi")  # 50.32h over 17140 clips
    avg = twi.hours * 3600 / twi.clips                    # ~10.6s/clip
    # 30-minute budget -> roughly 1800/avg clips, capped to availability
    plan = plan_samples([twi], per_language=None, max_seconds=1800, split="train")
    p = plan[0]
    assert p["planned"] == max(1, round(1800 / avg))
    assert abs(p["planned_hours"] - 0.5) < 0.1     # close to the 0.5-hour target
    # clip cap also applied -> whichever is smaller wins
    plan2 = plan_samples([twi], per_language=50, max_seconds=1800, split="train")
    assert plan2[0]["planned"] == 50


def test_duration_cutoff_closest():
    from afrispeech_selector.builder import _duration_cutoff
    lengths = [10.0, 10.0, 10.0, 10.0]
    assert _duration_cutoff(lengths, 25) == 3   # 30 (k=3) vs 20 (k=2): |30-25|==|20-25|, keep crossing
    assert _duration_cutoff(lengths, 21) == 2   # 20 closer to 21 than 30
    assert _duration_cutoff(lengths, 100) == 4  # budget exceeds total -> take all
    assert _duration_cutoff(lengths, 0) == 0


def test_plan_samples_caps_to_available():
    cat = load_catalog()
    small = next(e for e in cat if e.train < 100)  # e.g. luvale 20
    plan = plan_samples([small], per_language=1000, split="train")
    assert plan[0]["planned"] == small.train  # capped at availability
    plan2 = plan_samples([small], per_language=5, split="train")
    assert plan2[0]["planned"] == 5
