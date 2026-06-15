"""Command-line interface for AfriSpeech Selector.

The robust path for building training sets: select languages, pull a sized
sample, and export or push — all from the terminal, so a long download never
loses a browser connection. Examples:

    # Preview the top 10 country-balanced languages (no download)
    afrispeech-select --top 10 --max-per-country 2 --dry-run

    # Build 200 clips/language from the top 10, save to ./data
    afrispeech-select --top 10 --per-language 200 --out ./data

    # 30 min/language, only 3–20s clips, push to your HF repo
    afrispeech-select --top 10 --max-hours-per-lang 0.5 \\
        --min-clip-sec 3 --max-clip-sec 20 --push me/my-subset

    # Hand-pick specific languages
    afrispeech-select --languages twi_twi,hausa_hau --per-language 500 --out ./data
"""

from __future__ import annotations

import argparse
import sys

from .catalog import COUNTRY_NAMES, load_catalog
from .selector import filter_catalog, plan_samples, select_top


def _fmt_plan(plan: list[dict]) -> str:
    cols = [("language", 22), ("country_name", 14), ("iso", 6),
            ("hours", 7), ("available", 10), ("planned", 9), ("planned_hours", 8)]
    head = {"country_name": "country", "available": "in_split",
            "planned": "to_pull", "planned_hours": "est_h"}
    lines = ["  ".join(f"{head.get(c,c):<{w}}" for c, w in cols)]
    lines.append("  ".join("-" * w for _, w in cols))
    for p in plan:
        lines.append("  ".join(f"{str(p.get(c,'')):<{w}}"[:w] for c, w in cols))
    return "\n".join(lines)


def _resolve_languages(arg: str) -> list[str]:
    cat = {e.subset: e for e in load_catalog()}
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in cat:
            sys.exit(f"Unknown language/subset '{tok}'. Use --list-langs to see options.")
        out.append(tok)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="afrispeech-select",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sel = p.add_argument_group("selection")
    sel.add_argument("--top", type=int, help="select the top-N languages by hours")
    sel.add_argument("--languages", help="comma-separated subset names (hand-pick mode)")
    sel.add_argument("--proportional", dest="proportional", action="store_true", default=True,
                     help="balance across countries (default)")
    sel.add_argument("--no-proportional", dest="proportional", action="store_false",
                     help="pure hours ranking, ignore country balance")
    sel.add_argument("--max-per-country", type=int, help="cap languages per country")
    sel.add_argument("--min-hours", type=float, default=0.0, help="drop languages below this many hours")
    sel.add_argument("--max-hours", type=float, help="drop languages above this many hours")
    sel.add_argument("--min-clips", type=int, default=0, help="drop languages with fewer clips")
    sel.add_argument("--countries", help="comma-separated ISO country codes to restrict to (e.g. GH,NG)")
    sel.add_argument("--split", choices=["train", "val", "test", "all"], default="train")

    siz = p.add_argument_group("per-language sizing")
    siz.add_argument("--per-language", type=int, help="max clips per language")
    siz.add_argument("--max-hours-per-lang", type=float, help="duration budget per language, hours (e.g. 0.5)")
    siz.add_argument("--min-clip-sec", type=float, help="drop clips shorter than this many seconds")
    siz.add_argument("--max-clip-sec", type=float, help="drop clips longer than this many seconds")
    siz.add_argument("--seed", type=int, default=42)

    out = p.add_argument_group("output")
    out.add_argument("--out", default="afrispeech_selection",
                     help="output directory / base name (default: afrispeech_selection)")
    out.add_argument("--format", default="disk",
                     help="comma list of: disk,zip,parquet,csv (default: disk)")
    out.add_argument("--push", metavar="REPO_ID", help="push to this HF dataset repo (user/name)")
    out.add_argument("--private", dest="private", action="store_true", default=True)
    out.add_argument("--public", dest="private", action="store_false")
    out.add_argument("--token", help="HF token (else uses HF_TOKEN env / cached login)")

    misc = p.add_argument_group("misc")
    misc.add_argument("--streaming", dest="streaming", action="store_true", default=None,
                      help="force streaming pull (default: on when a per-language limit is set)")
    misc.add_argument("--no-streaming", dest="streaming", action="store_false")
    misc.add_argument("--allow-full", action="store_true",
                      help="permit an uncapped pull (downloads whole shards, ~65 GB)")
    misc.add_argument("--dry-run", action="store_true", help="print the selection plan and exit")
    misc.add_argument("--list-langs", action="store_true", help="print the full catalog and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_langs:
        for e in sorted(load_catalog(), key=lambda x: (-x.hours)):
            print(f"{e.subset:<28} {e.language:<24} {e.country} "
                  f"{COUNTRY_NAMES.get(e.country, ''):<22} {e.hours:>6.1f}h  {e.clips} clips")
        return 0

    # ---- selection -------------------------------------------------------- #
    if args.languages:
        subsets = _resolve_languages(args.languages)
        cat = {e.subset: e for e in load_catalog()}
        chosen = [cat[s] for s in subsets]
    elif args.top:
        pool = filter_catalog(
            min_hours=args.min_hours, max_hours=args.max_hours, min_clips=args.min_clips,
            countries=args.countries.split(",") if args.countries else None,
            split=args.split,
        )
        if not pool:
            sys.exit("No languages match these filters. Loosen the hours/clip limits.")
        chosen = select_top(pool, args.top, proportional=args.proportional,
                            max_per_country=args.max_per_country)
    else:
        sys.exit("Specify either --top N or --languages a,b,c (see --help).")

    cap = args.per_language
    secs = args.max_hours_per_lang * 3600 if args.max_hours_per_lang else None
    plan = plan_samples(chosen, per_language=cap, max_seconds=secs,
                        min_clip_seconds=args.min_clip_sec, max_clip_seconds=args.max_clip_sec,
                        split=args.split)

    n_countries = len({p["country"] for p in plan})
    total_clips = sum(p["planned"] for p in plan)
    total_hours = round(sum(p.get("planned_hours", 0) for p in plan), 2)
    print(f"\nSelected {len(chosen)} languages across {n_countries} countries "
          f"(~{total_clips} clips / ~{total_hours} h, split={args.split}):\n", file=sys.stderr)
    print(_fmt_plan(plan))
    print("", file=sys.stderr)

    if args.dry_run:
        return 0

    # ---- guard against an accidental 65 GB pull --------------------------- #
    if cap is None and secs is None and not args.allow_full:
        sys.exit("No per-language limit set — that pulls whole shards (~65 GB). "
                 "Set --per-language or --max-hours-per-lang, or pass --allow-full.")

    use_streaming = args.streaming if args.streaming is not None else (cap is not None or secs is not None)

    # Import the heavy bits only now (keeps --help/--dry-run fast and offline).
    from .builder import build_dataset
    from . import export as _export

    def _progress(msg):
        print(f"  {msg}", file=sys.stderr, flush=True)

    ds = build_dataset(
        [e.subset for e in chosen], split=args.split, per_language=cap, max_seconds=secs,
        min_clip_seconds=args.min_clip_sec, max_clip_seconds=args.max_clip_sec,
        seed=args.seed, token=args.token, streaming=use_streaming, progress=_progress,
    )

    import os
    out_path = args.out.rstrip("/")
    out_dir = os.path.dirname(out_path) or "."
    base = os.path.basename(out_path) or "afrispeech_selection"

    fmts = {f.strip() for f in args.format.split(",") if f.strip()}
    written = []
    if "disk" in fmts:
        ds.save_to_disk(out_path)
        written.append(out_path + "/  (load_from_disk)")
    if "zip" in fmts:
        written.append(_export.export_archive(ds, out_dir=out_dir, name=base))
    if "parquet" in fmts:
        written.append(_export.export_parquet(ds, out_dir=out_dir, name=base))
    if "csv" in fmts:
        written.append(_export.export_metadata_csv(ds, out_dir=out_dir, name=base))

    print(f"\n✅ Built {len(ds)} clips, {len(chosen)} languages. "
          f"Columns: {', '.join(ds.column_names)}", file=sys.stderr)
    for w in written:
        print(f"   wrote {w}", file=sys.stderr)

    if args.push:
        url = _export.push_to_hub(ds, args.push, token=args.token, private=args.private,
                                  split=args.split)
        print(f"   pushed → {url}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
