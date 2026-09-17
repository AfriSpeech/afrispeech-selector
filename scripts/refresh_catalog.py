#!/usr/bin/env python3
"""Rebuild afrispeech_selector/data/catalog.tsv from all source datasets.

The catalog is the library's static strength table (hours = ranking signal).
This script merges every supported source corpus into one TSV with a ``dataset``
column:

    afrispeech    AfriSpeech/african-speech-public_v1          (142 configs)
    waxal         google/WaxalNLP                              ASR + TTS configs
    omnilingual   facebook/omnilingual-asr-corpus              African configs only
    open-bible    AfriSpeech/open-bible-speech-african         (19 configs)

It pulls per-split row counts and, where the corpus has a duration column,
samples it to estimate hours — no audio downloads, so it runs in a couple of
minutes. WaxalNLP has no duration column, so its per-config hours are
apportioned from the published corpus totals (1,250 h ASR / 180 h TTS).

Language names / ISO codes / countries are resolved through the afriso
curated list of African languages (github.com/AfriSpeech/afriso), with a few
per-corpus overrides where a corpus uses non-standard tags.

Usage:
    python scripts/refresh_catalog.py                      # rebuild all sources
    python scripts/refresh_catalog.py --sources waxal,open-bible --out /tmp/cat.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from afrispeech_selector.catalog import DATASETS, load_catalog  # noqa: E402

API = "https://datasets-server.huggingface.co"
AFRISO_LANGS_URL = (
    "https://raw.githubusercontent.com/AfriSpeech/afriso/main/src/afriso/data/languages.csv"
)

FIELDS = ["subset", "language", "iso", "country", "clips", "hours", "train", "val", "test", "dataset"]


# --- tiny HTTP helper with retries ----------------------------------------- #
_io_lock = threading.Lock()
_io_last = 0.0          # monotonic timestamp of last request start
_IO_MIN_GAP = 0.8       # seconds between request starts (stays under server quota)

_cache_dir = Path.home() / ".cache" / "afrispeech-selector"
_cache_file = _cache_dir / "refresh.json"
_cache: dict = {}


def _load_cache() -> dict:
    global _cache
    if _cache:
        return _cache
    try:
        _cache = json.loads(_cache_file.read_text())
    except (OSError, ValueError):
        _cache = {}
    return _cache


def _save_cache() -> None:
    try:
        _cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_file.write_text(json.dumps(_cache))
    except OSError as e:  # noqa: BLE001
        print(f"  warn: cache write failed: {e}", file=sys.stderr)


def _cached(name: str):
    return _load_cache().get(name)


def _cache_set(name: str, value) -> None:
    _load_cache()[name] = value
    _save_cache()


def _request(url: str) -> dict:
    global _io_last
    with _io_lock:
        now = time.monotonic()
        gap = _IO_MIN_GAP - (now - _io_last)
        if gap > 0:
            time.sleep(gap)
        _io_last = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": "afrispeech-selector"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())


def _get_json(url: str, tries: int = 8) -> dict:
    last = None
    for i in range(tries):
        try:
            return _request(url)
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429 or e.code >= 500:
                wait = 1.5 * (2 ** i)          # 1.5s, 3s, 6s, 12s, ...
                print(f"  rows {e.code} -> retry quiesced {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
            else:
                break
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.6 * (i + 1))
    raise RuntimeError(f"request failed: {url} :: {last}")


def _run_in_parallel(jobs: list[dict], workers: int = 8) -> None:
    """Run ``jobs`` (each {"fn": callable, "what": str}) concurrently.

    The return value / exception are stored on each job dict under "result" /
    "error" so callers can regroup afterwards. HTTP traffic is globally gated
    by ``_request`` so worker count only overlaps CPU/parse time.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(j["fn"]): j for j in jobs}
        for fut in list(futs):
            j = futs[fut]
            try:
                j["result"] = fut.result()
            except Exception as e:  # noqa: BLE001
                j["error"] = e
                print(f"  error on {j.get('what', '?')}: {e}", file=sys.stderr)


def _fetch_rows(dataset: str, config: str, split: str,
                n: int = 10) -> tuple[int, list[float]]:
    """Rows call that returns (num_rows_total, sampled duration values)."""
    cache_key = f"rows:{dataset}/{config}/{split}"
    hit = _cached(cache_key)
    if hit is not None:
        return 0, hit
    url = (f"{API}/rows?dataset={urllib.parse.quote(dataset)}"
           f"&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}"
           f"&offset=0&length={n}")
    d = _get_json(url)
    durs = []
    for r in d.get("rows", []):
        row = r.get("row") or {}
        for col in ("duration", "duration_seconds"):
            v = row.get(col)
            if v is not None:
                try:
                    durs.append(float(v))
                except (TypeError, ValueError):
                    pass
                break
    _cache_set(cache_key, durs)
    return int(d.get("num_rows_total") or 0), durs


def size_by_split(dataset: str) -> dict[str, dict[str, int]]:
    """/size call -> {config: {split: num_rows}}, cached per dataset."""
    key = f"size:{dataset}"
    hit = _cached(key)
    if hit is not None:
        return hit
    d = _get_json(f"{API}/size?dataset={urllib.parse.quote(dataset)}")
    size = d.get("size") or {}
    out: dict[str, dict[str, int]] = {}
    for s in size.get("splits") or []:
        n = s.get("num_rows")
        if n is None:
            n = s.get("estimated_num_rows") or 0
        out.setdefault(s["config"], {})[s["split"]] = int(n)
    _cache_set(key, out)
    return out


# --- afriso language metadata ---------------------------------------------- #
def _split_countries(value: str) -> list[str]:
    return [c for c in re.split(r"[;,]\s*", value or "") if c]


class AfriSo:
    """Curated African-language metadata (github.com/AfriSpeech/afriso)."""

    def __init__(self, rows: list[dict]):
        self.by3: dict[str, dict] = {}
        self.by2: dict[str, dict] = {}
        for r in rows:
            code3 = r.get("iso639_3", "").strip()
            code2 = r.get("iso639_1", "").strip()
            countries = [c for c in _split_countries(r.get("countries", "")) if c]
            info = {"name": r.get("name", "").strip(), "iso": code3, "countries": countries}
            if code3:
                self.by3[code3] = info
            if code2:
                self.by2[code2] = info

    def get(self, code: str, from_2: bool = False) -> dict | None:
        if code in self.by3:
            return self.by3[code]
        return self.by2.get(code) if from_2 else None

    def primary_country(self, code: str, from_2: bool = False, fallback: str = "") -> str:
        info = self.get(code, from_2=from_2)
        if info and info["countries"]:
            return info["countries"][0]
        return fallback

    def name(self, code: str, from_2: bool = False, fallback: str = "") -> str:
        info = self.get(code, from_2=from_2)
        return info["name"] if info and info["name"] else fallback


def load_afriso(path: Path | None = None) -> AfriSo:
    if path and path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    else:
        text = ""
        last = None
        for _ in range(4):
            try:
                text = urllib.request.urlopen(AFRISO_LANGS_URL, timeout=60).read().decode()
                break
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1)
        if not text:
            raise RuntimeError(f"could not fetch {AFRISO_LANGS_URL} :: {last}")
        rows = list(csv.DictReader(text.splitlines()))
    return AfriSo(rows)


# --- rows ------------------------------------------------------------------- #
def _row(subset, language, iso, country, clips, hours, train, val, test, dataset) -> dict:
    return dict(subset=subset, language=language, iso=iso, country=country,
                clips=int(clips), hours=round(float(hours), 2),
                train=int(train), val=int(val), test=int(test), dataset=dataset)


def build_waxal(afri: AfriSo) -> list[dict]:
    # Google tags that aren't valid ISO 639-3, plus representative countries.
    OVERRIDES = {"mas": ("Masaaba", "myx", "UG"), "sog": ("Soga", "xog", "UG"),
                 "bau": ("Baoulé", "bci", "CI")}
    COUNTRY = {
        "ach": "UG", "aka": "GH", "amh": "ET", "dag": "GH", "dga": "GH", "ewe": "GH",
        "ful": "NG", "hau": "NG", "ibo": "NG", "kik": "KE", "kpo": "GH", "lin": "CD",
        "lug": "UG", "luo": "KE", "mlg": "MG", "nyn": "UG", "orm": "ET", "pcm": "NG",
        "sid": "ET", "sna": "ZW", "swa": "KE", "tir": "ER", "twi": "GH", "wal": "ET",
        "yor": "NG", "fat": "GH",
    }
    # Published corpus totals (WaxalNLP card); per-config hours are apportioned
    # proportionally to each config's share of labelled clips.
    TARGET = {"asr": 1250.0, "tts": 180.0}

    counts = size_by_split(DATASETS["waxal"])
    # config -> (code, kind), skipping the "_v2" speaker-disjoint re-splits
    # (same audio, would double-count a language).
    configs: dict[str, tuple[str, str]] = {}
    for cfg in counts:
        code, sep, kind = cfg.rpartition("_")
        if sep and kind in ("asr", "tts") and not code.endswith("_v2"):
            configs[cfg] = (code, kind)

    labelled: dict[tuple[str, str], dict[str, int]] = {
        (code, kind): {"train": 0, "val": 0, "test": 0}
        for cfg, (code, kind) in configs.items()
    }
    for cfg, (code, kind) in configs.items():
        for sp, n in counts[cfg].items():
            if sp == "unlabeled":
                continue
            labelled[(code, kind)]["val" if sp == "validation" else sp] = n

    label_total = {kind: 0 for kind in ("asr", "tts")}
    for (code, kind), c in labelled.items():
        label_total[kind] += sum(c.values())

    rows = []
    for (code, kind), c in sorted(labelled.items()):
        cfg = next(cf for cf, (co, ki) in configs.items() if (co, ki) == (code, kind))
        name, iso, country = OVERRIDES.get(
            code, (afri.name(code, fallback=code), code,
                  COUNTRY.get(code, afri.primary_country(code, fallback=""))))
        clips = sum(c.values())
        total = label_total[kind] or 1
        hours = TARGET[kind] * clips / total
        rows.append(_row(cfg, name, iso, country, clips, hours,
                         c["train"], c["val"], c["test"], "waxal"))
    return rows


def build_omnilingual(afri: AfriSo) -> list[dict]:
    counts = size_by_split(DATASETS["omnilingual"])
    african = [cfg for cfg in counts if cfg != "default" and afri.get(cfg.partition("_")[0])]

    # Sample durations from an evenly-spread subset of configs (~60) and use
    # one pooled mean for all configs — recordings are homogeneous short
    # phrases, so per-language mean-vs-global differ by only a few seconds.
    spread = max(1, len(african) // 60)
    jobs = []
    for cfg in african[::spread]:
        code, _, _ = cfg.partition("_")
        sp = next((s for s in ("train", "dev", "test") if s in counts[cfg]), None)
        if not sp:
            continue
        jobs.append({"what": f"omni::{cfg}",
                     "fn": (lambda d=DATASETS["omnilingual"], c=cfg, s=sp:
                            _fetch_rows(d, c, s, n=8))})
    _run_in_parallel(jobs)

    # Pool sampled durations; the mean is the right estimator for total hours,
    # but cap absurd values (> 120 s) that would skew it.
    durations: list[float] = []
    for j in jobs:
        dur = j["result"][1] if ("error" not in j and j.get("result")) else []
        durations.extend(d for d in dur if d <= 120.0)
    global_mean = (sum(durations) / len(durations)) if durations else 0.0
    if not durations:
        print("  warn: omnilingual duration sample empty", file=sys.stderr)

    rows = []
    for cfg in african:
        code, _, script = cfg.partition("_")
        info = afri.get(code)
        if not info:
            continue
        name = info["name"] or code
        if script and script != "Latn":
            name = f"{name} ({script} script)"
        country = info["countries"][0] if info["countries"] else ""

        c = counts[cfg]
        train = c.get("train", 0); dev = c.get("dev", 0); test = c.get("test", 0)
        clips = train + dev + test
        hours = clips * global_mean / 3600 if clips else 0.0
        rows.append(_row(cfg, name, code, country, clips, hours,
                         train, dev, test, "omnilingual"))
    return rows


def build_open_bible(afri: AfriSo) -> list[dict]:
    # config name -> (ISO 639-1 tag, fallback country); resolved via afriso.
    ISO1 = {
        "Chichewa": ("ny", "MW"), "Dawro": ("dwr", "ET"), "Dholuo": ("luo", "KE"),
        "Ewe": ("ee", "GH"), "Gamo": ("gmv", "ET"), "Gofa": ("gof", "ET"),
        "Hausa": ("ha", "NG"), "Igbo": ("ig", "NG"), "Kikuyu": ("ki", "KE"),
        "Lingala": ("ln", "CD"), "Luganda": ("lg", "UG"), "Matengo": ("mgv", "TZ"),
        "Ndebele": ("nd", "ZW"), "Oromo": ("om", "ET"), "Shona": ("sn", "ZW"),
        "Swahili": ("sw", "TZ"), "Twi (Akuapem)": ("tw", "GH"), "Twi (Asante)": ("tw", "GH"),
        "Yoruba": ("yo", "NG"),
    }
    counts = size_by_split(DATASETS["open-bible"])

    jobs = []
    for cfg in counts:
        if "train" not in counts[cfg]:
            continue
        jobs.append({"what": f"bible::{cfg}",
                     "fn": (lambda d=DATASETS["open-bible"], c=cfg:
                            _fetch_rows(d, c, "train", n=8))})
    _run_in_parallel(jobs)
    by_cfg = {j["what"].split("::", 1)[1]: j for j in jobs}

    rows = []
    for cfg in counts:
        tag, fallback_country = ISO1[cfg]
        info = afri.get(tag, from_2=True)
        name = info["name"] if info else cfg
        iso = info["iso"] if info else (tag if len(tag) == 3 else "")
        country = info["countries"][0] if (info and info["countries"]) else fallback_country
        c = counts[cfg]
        train = c.get("train", 0); test = c.get("test", 0)
        clips = train + test
        j = by_cfg.get(cfg)
        dur = j["result"][1] if (j and "error" not in j and j.get("result")) else []
        mean = (sum(dur) / len(dur)) if dur else 0.0
        hours = clips * mean / 3600 if clips else 0.0
        rows.append(_row(cfg, name, iso, country, clips, hours, train, 0, test, "open-bible"))
    return rows


def build_afrispeech() -> list[dict]:
    """Carry over the existing (exact) AfriSpeech rows unchanged."""
    return [_row(e.subset, e.language, e.iso, e.country, e.clips, e.hours,
                 e.train, e.val, e.test, "afrispeech")
            for e in load_catalog() if e.dataset == "afrispeech"]


# --- main ------------------------------------------------------------------ #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="afrispeech,waxal,omnilingual,open-bible",
                    help="comma-separated catalogue keys to rebuild")
    ap.add_argument("--out", default=str(ROOT / "afrispeech_selector" / "data" / "catalog.tsv"))
    ap.add_argument("--afriso-langs", default=None,
                    help="local afriso languages.csv (default: fetch from GitHub)")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    print(f"existing catalog: {len(load_catalog())} rows", file=sys.stderr)

    afri = load_afriso(Path(args.afriso_langs) if args.afriso_langs else None)
    print(f"afriso: {len(afri.by3)} African language codes", file=sys.stderr)

    builders = {
        "afrispeech": build_afrispeech,
        "waxal": build_waxal,
        "omnilingual": build_omnilingual,
        "open-bible": build_open_bible,
    }
    all_rows: list[dict] = []
    for key in sources:
        fn = builders.get(key)
        if fn is None:
            print(f"skip unknown source: {key}", file=sys.stderr)
            continue
        rows = fn(afri) if key != "afrispeech" else fn()
        h = round(sum(r["hours"] for r in rows), 1)
        clips = sum(r["clips"] for r in rows)
        print(f"  {key}: {len(rows)} rows, {clips} clips, {h} h", file=sys.stderr)
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (-r["hours"], r["language"], r["subset"]))
    seen, final = set(), []
    for r in all_rows:
        k = (r["dataset"], r["subset"])
        if k in seen:
            continue
        seen.add(k)
        final.append(r)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(final)
    print(f"Wrote {len(final)} rows -> {args.out}", file=sys.stderr)
    print(f"Total: {sum(r['clips'] for r in final)} clips, "
          f"{round(sum(r['hours'] for r in final), 1)} h (estimated).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())