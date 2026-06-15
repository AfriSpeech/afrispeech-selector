"""Tests for the CLI arg parsing / selection and the UI command builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from afrispeech_selector import cli


def test_parser_top_and_sizing():
    args = cli.build_parser().parse_args(
        ["--top", "10", "--max-per-country", "2", "--per-language", "200",
         "--max-hours-per-lang", "0.5", "--min-clip-sec", "3", "--max-clip-sec", "20"]
    )
    assert args.top == 10 and args.max_per_country == 2
    assert args.per_language == 200 and args.max_hours_per_lang == 0.5
    assert args.min_clip_sec == 3 and args.max_clip_sec == 20
    assert args.proportional is True
    assert cli.build_parser().parse_args(["--no-proportional"]).proportional is False


def test_resolve_languages_valid_and_invalid():
    assert cli._resolve_languages("twi_twi, hausa_hau") == ["twi_twi", "hausa_hau"]
    try:
        cli._resolve_languages("not_a_lang")
    except SystemExit as e:
        assert "Unknown" in str(e)
    else:
        raise AssertionError("expected SystemExit on unknown language")


def test_dry_run_exits_zero(capsys):
    rc = cli.main(["--top", "3", "--per-language", "50", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr()
    assert "to_pull" in out.out  # the plan table printed


def test_total_hours_splits_evenly(capsys):
    rc = cli.main(["--top", "8", "--max-per-country", "2", "--total-hours", "24", "--dry-run"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "÷ 8 languages = 3.00 h each" in err  # 24h / 8 langs


def test_uncapped_pull_blocked():
    try:
        cli.main(["--top", "3"])
    except SystemExit as e:
        assert "65 GB" in str(e) or "limit" in str(e)
    else:
        raise AssertionError("expected SystemExit guarding an uncapped pull")


def test_command_builder_matches_flags():
    import app
    cmd = app._cmd(mode="top", top=10, proportional=False, max_pc=2, min_h=10,
                   per_language=200, max_hours_per_lang=0.5, min_len=3, max_len=20,
                   split="train", out="data", fmt="disk,parquet", push="me/x")
    assert cmd.startswith("afrispeech-select --top 10")
    assert "--no-proportional" in cmd
    assert "--max-per-country 2" in cmd
    assert "--per-language 200" in cmd and "--max-hours-per-lang 0.5" in cmd
    assert "--min-clip-sec 3" in cmd and "--max-clip-sec 20" in cmd
    assert "--format disk,parquet" in cmd and "--push me/x" in cmd

    pick = app._cmd(mode="pick", languages=["twi_twi", "ga_gaa"], split="train",
                    per_language=100, out="afrispeech_selection")
    assert "--languages twi_twi,ga_gaa" in pick
    # default format 'disk' is omitted (it's the CLI default)
    assert "--format" not in pick
