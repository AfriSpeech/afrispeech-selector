"""AfriSpeech Selector — selection helper UI (optional).

A thin local browser UI for *exploring and selecting* languages. It does NOT
download audio itself — it previews your selection and emits the exact
``afrispeech-select`` command to run in your terminal, where the (possibly long)
download happens reliably. The CLI is the workhorse; this is just a builder.

Run:  python app.py   (then open http://127.0.0.1:7860)
"""

from __future__ import annotations

import os

import pandas as pd
import gradio as gr

from afrispeech_selector import (
    DATASET_ID,
    countries,
    filter_catalog,
    load_catalog,
    plan_samples,
    select_top,
)
from afrispeech_selector.catalog import COUNTRY_NAMES

CATALOG = load_catalog()
SPLITS = ["train", "val", "test", "all"]
LANG_LABELS = {
    f"{e.language} — {e.country_name} ({e.subset}, {e.hours:.1f}h)": e.subset
    for e in sorted(CATALOG, key=lambda x: x.language)
}
COUNTRY_CHOICES = [f"{c} — {COUNTRY_NAMES.get(c, c)}" for c in countries()]


def _code(c: str) -> str:
    return c.split(" — ")[0]


def _plan_to_df(plan: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(plan)
    cols = ["language", "country_name", "iso", "hours", "clips",
            "available", "planned", "planned_hours", "subset"]
    df = df[[c for c in cols if c in df.columns]]
    return df.rename(columns={"country_name": "country", "available": "in_split",
                              "planned": "to_pull", "planned_hours": "est_hours"})


def _cmd(*, mode, top=None, languages=None, proportional=True, max_pc=None,
         min_h=0, max_h=None, min_c=0, restrict=None, split="train",
         per_language=None, max_hours_per_lang=None, min_len=None, max_len=None,
         out="afrispeech_selection", fmt="disk", push=None) -> str:
    p = ["afrispeech-select"]
    if mode == "top":
        p += ["--top", str(int(top))]
        if not proportional:
            p += ["--no-proportional"]
        if max_pc:
            p += ["--max-per-country", str(int(max_pc))]
        if min_h:
            p += ["--min-hours", f"{float(min_h):g}"]
        if max_h:
            p += ["--max-hours", f"{float(max_h):g}"]
        if min_c:
            p += ["--min-clips", str(int(min_c))]
        if restrict:
            p += ["--countries", ",".join(_code(c) for c in restrict)]
    else:
        p += ["--languages", ",".join(languages)]
    p += ["--split", split]
    if per_language:
        p += ["--per-language", str(int(per_language))]
    if max_hours_per_lang:
        p += ["--max-hours-per-lang", f"{float(max_hours_per_lang):g}"]
    if min_len:
        p += ["--min-clip-sec", f"{float(min_len):g}"]
    if max_len:
        p += ["--max-clip-sec", f"{float(max_len):g}"]
    p += ["--out", out or "afrispeech_selection"]
    if fmt and fmt != "disk":
        p += ["--format", fmt]
    if push:
        p += ["--push", push]
    return " ".join(p)


def _secs(max_hours_per_lang):
    return float(max_hours_per_lang) * 3600 if max_hours_per_lang else None


def _render(chosen, *, split, per_language, max_hours_per_lang, min_len, max_len, command):
    cap = int(per_language) if per_language else None
    secs = _secs(max_hours_per_lang)
    lo = float(min_len) if min_len else None
    hi = float(max_len) if max_len else None
    plan = plan_samples(chosen, per_language=cap, max_seconds=secs,
                        min_clip_seconds=lo, max_clip_seconds=hi, split=split)
    total_clips = sum(p["planned"] for p in plan)
    total_hours = round(sum(p.get("planned_hours", 0) for p in plan), 2)
    n_countries = len({p["country"] for p in plan})
    msg = (f"**{len(chosen)}** languages across **{n_countries}** countries — "
           f"~**{total_clips}** clips / **~{total_hours} h** (split=`{split}`). "
           f"Run the command below in your terminal to download.")
    return _plan_to_df(plan), msg, command


def preview_top(n, proportional, max_pc, min_h, max_h, min_c, restrict, split,
                per_language, max_hours_per_lang, min_len, max_len, out, fmt, push):
    pool = filter_catalog(
        min_hours=float(min_h or 0), max_hours=float(max_h) if max_h else None,
        min_clips=int(min_c or 0),
        countries=[_code(c) for c in restrict] if restrict else None, split=split,
    )
    if not pool:
        return None, "No languages match these filters. Loosen the hours/clip limits.", ""
    chosen = select_top(pool, int(n), proportional=bool(proportional),
                        max_per_country=int(max_pc) if max_pc else None)
    command = _cmd(mode="top", top=n, proportional=proportional, max_pc=max_pc,
                   min_h=min_h, max_h=max_h, min_c=min_c, restrict=restrict, split=split,
                   per_language=per_language, max_hours_per_lang=max_hours_per_lang,
                   min_len=min_len, max_len=max_len, out=out, fmt=fmt, push=push)
    return _render(chosen, split=split, per_language=per_language,
                   max_hours_per_lang=max_hours_per_lang, min_len=min_len, max_len=max_len,
                   command=command)


def preview_pick(picked, split, per_language, max_hours_per_lang, min_len, max_len, out, fmt, push):
    if not picked:
        return None, "Pick at least one language.", ""
    subsets = [LANG_LABELS[l] for l in picked]
    chosen = [e for e in CATALOG if e.subset in subsets]
    command = _cmd(mode="pick", languages=subsets, split=split,
                   per_language=per_language, max_hours_per_lang=max_hours_per_lang,
                   min_len=min_len, max_len=max_len, out=out, fmt=fmt, push=push)
    return _render(chosen, split=split, per_language=per_language,
                   max_hours_per_lang=max_hours_per_lang, min_len=min_len, max_len=max_len,
                   command=command)


with gr.Blocks(title="AfriSpeech Selector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"# 🌍 AfriSpeech Selector\n"
        f"Explore & select African languages from **`{DATASET_ID}`** (ranked by recorded "
        f"**hours**), then copy the **`afrispeech-select`** command and run it in your "
        f"terminal to download. The terminal does the heavy pull — reliable, resumable, "
        f"and pipeable into training. Output schema: `audio · text · language · country · length`."
    )

    with gr.Tab("🏆 Top languages across Africa"):
        with gr.Row():
            n = gr.Number(value=10, label="How many languages (top-N)", precision=0, minimum=1)
            split_a = gr.Dropdown(SPLITS, value="train", label="Split")
        with gr.Row():
            proportional = gr.Checkbox(value=True, label="Balance across countries (one per country first, then fill)")
            max_pc = gr.Number(label="Max languages per country (blank = no cap)", precision=0)
        with gr.Accordion("Strength filters", open=False):
            with gr.Row():
                min_h = gr.Number(value=0, label="Min hours", minimum=0)
                max_h = gr.Number(label="Max hours (blank = no cap)")
                min_c = gr.Number(value=0, label="Min clips", precision=0, minimum=0)
            restrict_c = gr.Dropdown(COUNTRY_CHOICES, label="Restrict to countries (optional)", multiselect=True)
        btn_top = gr.Button("Preview & build command", variant="primary")

    with gr.Tab("🎯 Pick specific languages"):
        picked = gr.Dropdown(list(LANG_LABELS), label="Languages", multiselect=True)
        split_b = gr.Dropdown(SPLITS, value="train", label="Split")
        btn_pick = gr.Button("Preview & build command", variant="primary")

    gr.Markdown("### Per-language sizing")
    with gr.Row():
        per_lang = gr.Number(label="Max samples per language (blank = all)", precision=0)
        max_hr_lang = gr.Number(label="Max hours per language (blank = none, e.g. 0.5)", minimum=0)
        min_len = gr.Number(label="Min sample length (sec)", minimum=0)
        max_len = gr.Number(label="Max sample length (sec)", minimum=0)
    gr.Markdown(
        "_Sample-length window is a precondition: out-of-range clips are skipped while "
        "picking, so the sample/hour target is filled from in-range clips only._"
    )

    gr.Markdown("### Output")
    with gr.Row():
        out_path = gr.Textbox(value="afrispeech_selection", label="Output path (--out)")
        fmt = gr.Dropdown(["disk", "zip", "parquet", "csv", "disk,parquet",
                           "disk,zip,csv"], value="disk", label="Format (--format)")
        push_repo = gr.Textbox(label="Push to HF repo (optional, --push)", placeholder="you/my-subset")

    gr.Markdown("### Selection")
    status = gr.Markdown()
    table = gr.Dataframe(label="What the command will pull", interactive=False, wrap=True)
    gr.Markdown("### ▶️ Run this in your terminal")
    command_box = gr.Code(label="command", language="shell")
    gr.Markdown(
        "_First time: `pip install -e .` (gives the `afrispeech-select` command), or use "
        "`python -m afrispeech_selector …`. Add `--push you/repo` and `--token …` to upload. "
        "`--dry-run` previews without downloading._"
    )

    common = [per_lang, max_hr_lang, min_len, max_len, out_path, fmt, push_repo]
    btn_top.click(
        preview_top,
        [n, proportional, max_pc, min_h, max_h, min_c, restrict_c, split_a, *common],
        [table, status, command_box],
    )
    btn_pick.click(
        preview_pick, [picked, split_b, *common], [table, status, command_box],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        share=os.environ.get("SHARE", "0") == "1",
        inbrowser=os.environ.get("NO_BROWSER", "0") != "1",
        show_error=True,
    )
