# 🌍 AfriSpeech Selector

Build training sets from
[`AfriSpeech/african-speech-public_v1`](https://huggingface.co/datasets/AfriSpeech/african-speech-public_v1)
from your terminal. Rank African languages by recorded **hours** (strength),
pick a country-balanced top-N (or hand-pick specific languages), size the sample,
and export locally or push to your own Hugging Face dataset repo — in a fixed,
training-ready schema.

It's a **CLI**: the heavy audio download runs in your terminal, so it's robust,
resumable, scriptable, and pipes straight into a training script. An optional
local browser UI helps you *explore and build the command* — but the CLI does
the work.

## Install

```bash
git clone https://github.com/AfriSpeech/afrispeech-selector.git
cd afrispeech-selector
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # gives you the `afrispeech-select` command
```

(Use `python3` — the code uses non-ASCII text and won't run under Python 2.)

## CLI usage

```bash
# Preview the top 10 country-balanced languages — no download
afrispeech-select --top 10 --max-per-country 2 --per-language 200 --dry-run

# Build it: 200 clips/language → ./data (HF load_from_disk format)
afrispeech-select --top 10 --max-per-country 2 --per-language 200 --out ./data

# 30 min/language, only 3–20s clips, also write a parquet file
afrispeech-select --top 10 --max-hours-per-lang 0.5 \
    --min-clip-sec 3 --max-clip-sec 20 --out ./data --format disk,parquet

# Hand-pick languages and push to your HF repo
afrispeech-select --languages twi_twi,hausa_hau --per-language 500 \
    --push you/my-subset --token "$HF_TOKEN"

# See every available language/subset
afrispeech-select --list-langs
```

No install? `python3 -m afrispeech_selector …` works the same from the repo.

### Key options

| flag | meaning |
|------|---------|
| `--top N` | select the top-N languages by hours |
| `--languages a,b,c` | hand-pick specific subsets instead |
| `--max-per-country N` | cap languages per country (balance) |
| `--no-proportional` | pure hours ranking, ignore country balance |
| `--min-hours / --max-hours / --min-clips` | filter the language pool by strength |
| `--countries GH,NG` | restrict to these countries |
| `--per-language N` | max clips per language |
| `--max-hours-per-lang H` | duration budget per language (decimals OK, e.g. `0.5`) |
| `--min-clip-sec / --max-clip-sec` | per-sample length window (out-of-range clips skipped while picking) |
| `--split train\|val\|test\|all` | which split to draw from |
| `--out PATH` | output directory / base name |
| `--format disk,zip,parquet,csv` | one or more output formats |
| `--push REPO_ID [--public] [--token …]` | push to an HF dataset repo |
| `--dry-run` | print the plan and stop |

Capped pulls **stream** from the Hub and only transfer the samples you ask for.
An uncapped "full build" downloads whole shards (the dataset is ~65 GB) and must
be enabled with `--allow-full`.

### Output schema

| column | meaning |
|--------|---------|
| `audio` | decoded waveform (HF `Audio`: `array` + `sampling_rate`) |
| `text` | transcription |
| `language` | language label |
| `country` | ISO 3166-1 alpha-2 code |
| `length` | clip duration in seconds |
| `iso`, `subset` | ISO 639-3 code and source config (traceability) |

Load a result later:

```python
from datasets import load_from_disk
ds = load_from_disk("data")                    # from --format disk
# or: Dataset.from_parquet("data.parquet")
ds = ds.train_test_split(test_size=0.1)        # feed your trainer
```

## Optional: the selection UI

A local browser helper for exploring languages and **building the command** (it
does not download — it hands you the `afrispeech-select` line to run):

```bash
pip install -e ".[ui]"     # adds gradio
python app.py              # opens http://127.0.0.1:7860
```

## Use as a library

```python
from afrispeech_selector import filter_catalog, select_top, build_dataset

pool  = filter_catalog(min_hours=10, split="train")
langs = select_top(pool, 10, proportional=True, max_per_country=2)
ds = build_dataset(langs, split="train", max_seconds=1800,
                   min_clip_seconds=3, max_clip_seconds=20, streaming=True)
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/        # selection, builder, and CLI tests (mostly offline)
```

## Keeping the catalog current

`data/catalog.tsv` is the static strength table (fast, offline). Regenerate it
when the source dataset changes:

```bash
python3 scripts/refresh_catalog.py --token "$HF_TOKEN"
```

## Project layout

```
afrispeech_selector/
  cli.py        the `afrispeech-select` command (workhorse)
  catalog.py    load the language table; country names
  selector.py   ranking, filtering, country-proportional top-N, plan
  builder.py    pull subsets from the Hub → standard schema (streaming)
  export.py     zip / parquet / manifest / push_to_hub
app.py          optional selection UI (emits the CLI command)
data/catalog.tsv  strength table (hours, clips, splits per subset)
scripts/        refresh_catalog.py, clean_source_dataset.py
tests/          selection, builder, CLI tests
```

## License

CC-BY-4.0
