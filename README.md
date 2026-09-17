# AfriSpeech Selector

**AfriSpeech Selector gives you access to 1,938,892 utterances · ~7,854 hours ·
308 African languages from 4 public corpora for training TTS and ASR models.**

Select languages by recorded **hours** (strength) — a country-balanced top-N or a
hand-picked set, sized the way you want — and get the **audio + metadata in the
format your training pipeline expects**. You take it from there; the tool doesn't
do any text normalisation or cleaning (that's your framework's job).

It's clean read speech with aligned transcripts, so the natural fit is **TTS** —
export WAVs + a manifest for **LJSpeech, Piper, VITS, or MeloTTS**. It works for
**ASR** too (`load_from_disk` / Parquet, or stream with `stream_dataset(...)`),
which is handy for supplementing low-resource languages.

The strength table is merged from four source corpora (see
**[Data sources](#data-sources)**): AfriSpeech, Google WaxalNLP,
Meta OmniLingual ASR, and Open Bible Speech (African). Use
`--dataset` to select into any one of them.

**Redistribution:** building a *local* working set for your own training is fine.
Redistributing copies of the audio (e.g. `--push` to a public repo) is **not
recommended**, given the permissions of the underlying public data sources.

## Available languages

**369 language configs · 1,938,892 clips · ~7,854 hours · 45 countries**, drawn
from 4 corpora. Hours is the strength signal used for ranking. The `--languages`
value is the name you pass to pick a language (e.g. `--languages twi_twi`,
`ajima_ajm`…). List them anytime with `afrispeech-select --list-langs`, or see
the full **[language catalog](#language-catalog)** at the bottom. To restrict the
pool to one source corpus pass `--dataset`, e.g.
`afrispeech-select --list-langs --dataset open-bible`.


## Install

```bash
git clone https://github.com/AfriSpeech/afrispeech-selector.git
cd afrispeech-selector
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # gives you the `afrispeech-select` command
```

(Use `python3` — the code uses non-ASCII text and won't run under Python 2.)

## Quickstart — one language, ready to train

Most people just want one language for **TTS**. Pick its name
(`afrispeech-select --list-langs`, or the catalog at the bottom) and run one
command. Clips are filtered to a **3–15 s** window by default, so the result is
training-ready.

```bash
# ~5 hours of Twi as an LJSpeech TTS dataset (wavs/ + metadata.csv), 22.05 kHz
afrispeech-select --languages twi_twi --total-hours 5 --out data/twi --format ljspeech

# …for Piper / VITS / MeloTTS instead, just change --format:
afrispeech-select --languages twi_twi --total-hours 5 --out data/twi --format piper
```

(Using it for ASR? See [Using it for ASR](#using-it-for-asr) below.)

That's it — `data/twi` now holds the audio + metadata in the right layout (omit
`--out` and it defaults to `data/<language>`). Want more or less? Change
`--total-hours` (or use `--per-language N` for a clip count).
Want longer/shorter clips? Set `--min-clip-sec` / `--max-clip-sec` (defaults 3 / 15).

**Asking for more than a language has?** You get **everything available** — the
hour/clip target is an upper bound; the tool never pads or repeats. When your
request can't be met, the CLI tells you how much actually exists and **asks you to
confirm** before downloading, e.g.:

```text
Only ~2.37 h is available vs ~20 h requested. Proceed with all available? [y/N]
```

Pass `-y`/`--yes` to skip the prompt (for scripts/notebooks), or `--dry-run` to
see the achievable amount up front. (The 3–15 s filter trims a language's usable
hours below its catalog total, so e.g. a 50 h language yields somewhat less.)

## Export one language for a TTS framework (WAVs + manifest)

TTS data-prep reads WAVs + a manifest from disk. Pick your framework with
`--format`: `ljspeech` (generic / Coqui), `piper`, `vits`, or `melo`. Audio is
16-bit mono WAV at `--target-sr` (default 22050); transcripts are written
**verbatim** — no normalisation.

```bash
# LJSpeech layout (wavs/ + metadata.csv: id|text|text)
afrispeech-select --languages twi_twi --total-hours 5 --out data/twi --format ljspeech

# Piper (metadata.csv: id|speaker|text)
afrispeech-select --languages twi_twi --total-hours 5 --out data/twi --format piper

# VITS (filelist.txt + speakers.txt) and/or MeloTTS (metadata.list)
afrispeech-select --languages twi_twi --total-hours 5 --out data/twi --format vits,melo
```

Each writes `<out>/wavs/*.wav` plus the manifest. Phonemisation / text cleaning
is left to the framework's own preprocessor.

No install? `python3 -m afrispeech_selector …` works the same from the repo.

## Using it for ASR

Being clean read speech with transcripts, it's well suited to **supplementing**
low-resource ASR (less so as a stand-alone set for spontaneous/noisy audio).
Export an on-disk 🤗 dataset, or stream it straight into a Trainer with no copy:

```bash
afrispeech-select --languages twi_twi --total-hours 5 --out data/twi \
    --format disk,parquet --target-sr 16000
```

```python
from datasets import load_from_disk
ds = load_from_disk("data/twi").train_test_split(test_size=0.1)

# …or stream (no local copy) — `afrispeech-select … --recipe` prints this:
from afrispeech_selector import stream_dataset
ds = stream_dataset(["twi_twi"], split="train", max_seconds=5 * 3600, target_sampling_rate=16000)
```

## Use it in a Colab / Kaggle training notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AfriSpeech/afrispeech-selector/blob/main/notebooks/afrispeech_selector.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/AfriSpeech/afrispeech-selector/blob/main/notebooks/afrispeech_selector.ipynb)

The example notebook is meant to be **dropped into your own training notebook**:
its cells install the tool and pull your selected language(s) **straight into the
running session** — write `wavs/` + a manifest for a TTS framework, or stream into
ASR training — so your training script has direct access to the speech data needed
for training.

## Selecting multiple languages

Everything above works for several languages too — the format/output flags are
identical, you just choose *which* languages. `--total-hours` is split evenly
across them (so each gets its fair share).

**1. Name the languages you want:**

```bash
# Ghanaian languages, 15 h total (5 h each), LJSpeech
afrispeech-select --languages twi_twi,ewe_ewe,ga_gaa --total-hours 15 \
    --out data/gh --format ljspeech
```

**2. Or pick across the dataset by strength / balance** — top-N by hours, with
optional country balancing and pool filters:

```bash
# Top 10 languages, at most 2 per country, 24 h total
afrispeech-select --top 10 --max-per-country 2 --total-hours 24 \
    --out data/multi --format ljspeech

# Narrow the pool first: only languages >= 20 h, only Ghana & Nigeria
afrispeech-select --top 8 --min-hours 20 --countries GH,NG --total-hours 16 \
    --out data/wa --format ljspeech
```

**Preview before pulling** — list what matches, or dry-run a selection:

```bash
afrispeech-select --list-langs --min-hours 20 --countries GH,NG   # what matches
afrispeech-select --top 10 --max-per-country 2 --dry-run          # what it would pick
```

## All options

| flag | meaning |
|------|---------|
| `--top N` | select the top-N languages by hours |
| `--languages a,b,c` | hand-pick specific subsets instead |
| `--max-per-country N` | cap languages per country (balance) |
| `--no-proportional` | pure hours ranking, ignore country balance |
| `--min-hours / --max-hours / --min-clips` | filter the language pool by strength |
| `--countries GH,NG` | restrict to these countries |
| `--dataset afrispeech,waxal,omnilingual,open-bible` | restrict to these source corpora (default: all). `afrispeech-select --list-langs --dataset waxal` shows them |
| `--total-hours H` | total audio across the selection, split evenly per language |
| `--per-language N` | max clips per language |
| `--max-hours-per-lang H` | duration budget per language (decimals OK, e.g. `0.5`) |
| `--min-clip-sec / --max-clip-sec` | per-sample length window, **default 3 / 15 s** (out-of-range clips skipped; `--min-clip-sec 0 --max-clip-sec 9999` to disable) |
| `--split train\|val\|test\|all` | which split to draw from |
| `--target-sr HZ` | resample audio (e.g. `16000` for ASR, `22050` for TTS) |
| `--schema asr\|whisper\|common_voice` | reshape columns for a training framework |
| `--recipe` | print a `stream_dataset(...)` snippet for this selection (no copy) |
| `--out PATH` | output directory / base name |
| `--format …` | HF: `disk,zip,parquet,csv` · TTS: `ljspeech,piper,vits,melo` |
| `--push REPO_ID [--public] [--token …]` | push to an HF dataset repo (creates a copy) |
| `--dry-run` / `--list-langs` | preview the plan / list matching languages |
| `-y` / `--yes` | skip the confirmation prompt when the request exceeds what's available |

Capped pulls **stream** from the Hub and only transfer the samples you ask for.
An uncapped "full build" downloads whole shards and must be enabled with
`--allow-full`.

Downloads work anonymously but the Hub rate-limits them; for faster, higher-rate
downloads set a free read token — `export HF_TOKEN=hf_…` (picked up automatically)
or pass `--token`.

## Output schema

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
ds = load_from_disk("data/twi")                  # from --format disk
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

One language:

```python
from afrispeech_selector import build_dataset, export_tts, stream_dataset

# Stream into training — no copy
ds = stream_dataset(["twi_twi"], split="train", max_seconds=5 * 3600, target_sampling_rate=16000)

# …or materialise and export for a TTS framework
copy = build_dataset(["twi_twi"], split="train", max_seconds=5 * 3600, streaming=True)
export_tts(copy, out_dir="./twi", fmt="ljspeech", sampling_rate=22050)
```

Several languages — name them, or select across the dataset:

```python
from afrispeech_selector import filter_catalog, select_top, stream_dataset

langs = select_top(filter_catalog(min_hours=20), 10, proportional=True, max_per_country=2)
ds = stream_dataset(langs, split="train", per_language=200, target_sampling_rate=16000)
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/        # selection, builder, and CLI tests (mostly offline)
```

## Keeping the catalog current

`afrispeech_selector/data/catalog.tsv` is the static strength table (fast,
offline; bundled in the package). Regenerate it when the source datasets change:

```bash
python3 scripts/refresh_catalog.py              # rebuild all four sources
python3 scripts/refresh_catalog.py --sources waxal,open-bible   # a subset
```

The script reads per-split row counts from the datasets-server API and estimates
hours from sampled durations (WaxalNLP has no duration column, so its hours are
apportioned from the published corpus totals). No audio is downloaded. Language
ISO codes / names / countries come from the curated afriso list of African
languages; responses are cached in `~/.cache/afrispeech-selector/` so re-runs
stay offline.

## Project layout

```
afrispeech_selector/
  cli.py        the `afrispeech-select` command (workhorse)
  catalog.py    load the language table; country names
  selector.py   ranking, filtering, country-proportional top-N, plan
  builder.py    pull subsets → standard schema; stream_dataset (no copy) + apply_schema
  export.py     HF (zip/parquet/manifest/push) + TTS (ljspeech/piper/vits/melo)
app.py          optional selection UI (emits the CLI command)
  data/catalog.tsv  strength table (4 source corpora, ~369 configs) — bundled in the package
scripts/        refresh_catalog.py
tests/          selection, builder, CLI tests
```

## Data sources

The catalog merges four public corpora of African speech. A `dataset` column in
`data/catalog.tsv` (surfaced as the `src` column in `--list-langs` and the
`--dataset` CLI filter) keeps each row traceable to its source.

| Corpus | House in catalog | Configs | Clips | Hours | Notes |
|--------|------------------|--------:|------:|------:|-------|
| [AfriSpeech](https://huggingface.co/datasets/AfriSpeech/african-speech-public_v1) | `afrispeech` | 142 | 778,520 | 2267.9 | Short, clean read speech; the original corpus (hours exact) |
| [Google WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) | `waxal` | 34 | 467,430 | 1430.0 | ASR (19) + TTS (15) configs; hours apportioned from published totals |
| [Meta OmniLingual ASR](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus) | `omnilingual` | 174 | 140,035 | 2417.9 | African configs only (matched via [afriso](https://github.com/AfriSpeech/afriso) to the curated African-language list); hours estimated from sampled clip durations |
| [Open Bible Speech (African)](https://huggingface.co/datasets/AfriSpeech/open-bible-speech-african) | `open-bible` | 19 | 552,907 | 1738.0 | African Bible readings; hours estimated from sampled clip durations |

Hours for WaxalNLP, OmniLingual and Open Bible are **estimates** derived from
row counts and sampled clip durations (WaxalNLP publishes no per-clip durations,
so it is split proportionally to its 1,250 h / 180 h card totals). The exact
download volume also depends on your `--min-clip-sec / --max-clip-sec` window.

Check each dataset's card on the Hub for license and reuse terms.

## Language catalog

All **369 language configs**, sorted by hours (strength), with the source corpus
in the last column. The `--languages` value is what you pass on the CLI; combine
with `--dataset` to restrict a selection to one source (e.g.
`afrispeech-select --top 10 --dataset omnilingual`).

| # | Language | ISO | Country | Hours | Clips | Train/Val/Test | `--languages` value | Source |
| 1 | Tigrinya | tir | Eritrea | 143.52 | 50,315 | 40,856/4,425/5,034 | `tir_asr` | waxal |
| 2 | Chichewa | nya | Malawi | 139.02 | 30,322 | 28,805/0/1,517 | `Chichewa` | open-bible |
| 3 | Wolaytta | wal | Ethiopia | 134.89 | 47,291 | 41,833/2,444/3,014 | `wal_asr` | waxal |
| 4 | Sidamo | sid | Ethiopia | 129.51 | 45,404 | 38,752/3,091/3,561 | `sid_asr` | waxal |
| 5 | Oromo | orm | Ethiopia | 128.49 | 45,045 | 38,185/3,078/3,782 | `orm_asr` | waxal |
| 6 | Amharic | amh | Ethiopia | 126.51 | 44,354 | 38,022/2,912/3,420 | `amh_asr` | waxal |
| 7 | Igbo | ibo | Nigeria | 119.02 | 30,011 | 28,510/0/1,501 | `Igbo` | open-bible |
| 8 | Gamo | gmv | Ethiopia | 118.78 | 30,200 | 28,690/0/1,510 | `Gamo` | open-bible |
| 9 | North Ndebele | nde | Botswana | 111.17 | 30,161 | 28,652/0/1,509 | `Ndebele` | open-bible |
| 10 | Lingala | lin | DR Congo | 108.68 | 28,790 | 27,350/0/1,440 | `Lingala` | open-bible |
| 11 | Kikuyu | kik | Kenya | 107.76 | 30,722 | 29,185/0/1,537 | `Kikuyu` | open-bible |
| 12 | Dawro | dwr | Ethiopia | 99.94 | 29,579 | 28,100/0/1,479 | `Dawro` | open-bible |
| 13 | Yoruba | yor | Benin | 95.84 | 30,625 | 29,093/0/1,532 | `Yoruba` | open-bible |
| 14 | Oromo | orm | Ethiopia | 93.68 | 30,413 | 28,892/0/1,521 | `Oromo` | open-bible |
| 15 | Swahili (macrolanguage) | swa | Burundi | 93.06 | 30,634 | 29,102/0/1,532 | `Swahili` | open-bible |
| 16 | Ganda | lug | Uganda | 91.11 | 30,440 | 28,918/0/1,522 | `Luganda` | open-bible |
| 17 | Ewe | ewe | Ghana | 86.24 | 30,164 | 28,655/0/1,509 | `Ewe` | open-bible |
| 18 | Hausa | hau | Burkina Faso | 83.64 | 30,717 | 29,181/0/1,536 | `Hausa` | open-bible |
| 19 | Shona | sna | Botswana | 82.42 | 30,685 | 29,150/0/1,535 | `Shona` | open-bible |
| 20 | Twi | twi | Ghana | 78.92 | 30,565 | 29,036/0/1,529 | `Twi (Asante)` | open-bible |
| 21 | Gofa | gof | Ethiopia | 78.76 | 30,388 | 28,868/0/1,520 | `Gofa` | open-bible |
| 22 | Fulah | ful | Nigeria | 68.12 | 23,881 | 19,132/2,358/2,391 | `ful_asr` | waxal |
| 23 | Luo (Kenya and Tanzania) | luo | Kenya | 67.36 | 30,429 | 28,907/0/1,522 | `Dholuo` | open-bible |
| 24 | Malagasy | mlg | Madagascar | 65.85 | 23,086 | 18,526/2,270/2,290 | `mlg_asr` | waxal |
| 25 | Malagasy | mlg | Madagascar | 61.32 | 20,287 | 18,900/746/641 | `malagasy_mlg` | afrispeech |
| 26 | Kabuverdianu | kea | Cabo Verde | 58.07 | 19,728 | 18,156/699/873 | `kabuverdianu_kea` | afrispeech |
| 27 | Twi | twi | Ghana | 55.28 | 30,270 | 28,756/0/1,514 | `Twi (Akuapem)` | open-bible |
| 28 | Southern Dagaare | dga | Ghana | 53.84 | 18,874 | 15,071/1,893/1,910 | `dga_asr` | waxal |
| 29 | Ewe | ewe | Ghana | 53.80 | 18,861 | 15,054/1,916/1,891 | `ewe_asr` | waxal |
| 30 | Shona | sna | Zimbabwe | 53.30 | 17,996 | 15,954/1,161/881 | `shona_sna` | afrispeech |
| 31 | Kabiye | kbp | Togo | 53.19 | 18,211 | 16,328/825/1,058 | `kabiye_kbp` | afrispeech |
| 32 | Eastern Yiddish (Hebr script) | ydd | South Africa | 51.80 | 3,000 | 1,950/300/750 | `ydd_Hebr` | omnilingual |
| 33 | Northern Betsimisaraka Malagasy | bmm | Madagascar | 51.76 | 2,998 | 2,998/0/0 | `bmm_Latn` | omnilingual |
| 34 | Orma | orc | Kenya | 51.76 | 2,998 | 1,500/900/598 | `orc_Latn` | omnilingual |
| 35 | Southern Betsimisaraka Malagasy | bzc | Madagascar | 51.75 | 2,997 | 2,997/0/0 | `bzc_Latn` | omnilingual |
| 36 | Tesaka Malagasy | tkg | Madagascar | 51.75 | 2,997 | 2,099/598/300 | `tkg_Latn` | omnilingual |
| 37 | Plateau Malagasy | plt | Madagascar | 51.71 | 2,995 | 2,995/0/0 | `plt_Latn` | omnilingual |
| 38 | Lingala | lin | DR Congo | 51.66 | 18,110 | 14,400/1,844/1,866 | `lin_asr` | waxal |
| 39 | Pular | fuf | Guinea | 51.51 | 2,983 | 2,983/0/0 | `fuf_Latn` | omnilingual |
| 40 | Egyptian Arabic (Arab script) | arz | Egypt | 51.45 | 2,980 | 2,384/298/298 | `arz_Arab` | omnilingual |
| 41 | Libyan Arabic (Arab script) | ayl | Egypt | 51.45 | 2,980 | 2,384/298/298 | `ayl_Arab` | omnilingual |
| 42 | Ikposo | kpo | Ghana | 51.40 | 18,020 | 14,415/1,760/1,845 | `kpo_asr` | waxal |
| 43 | Bagirmi Fulfulde | fui | Central African Republic | 51.18 | 2,964 | 2,964/0/0 | `fui_Latn` | omnilingual |
| 44 | Dagbani | dag | Ghana | 50.83 | 17,819 | 14,231/1,750/1,838 | `dag_asr` | waxal |
| 45 | Tunisian Arabic (Arab script) | aeb | Algeria | 50.50 | 2,925 | 2,925/0/0 | `aeb_Arab` | omnilingual |
| 46 | Twi | twi | Ghana | 50.32 | 17,140 | 15,794/828/518 | `twi_twi` | afrispeech |
| 47 | Shona | sna | Zimbabwe | 50.16 | 17,585 | 14,109/1,727/1,749 | `sna_asr` | waxal |
| 48 | Bassa (Cameroon) | bas | Cameroon | 48.58 | 16,269 | 14,981/620/668 | `bassa_cameroon_bas` | afrispeech |
| 49 | Masikoro Malagasy | msh | Madagascar | 46.57 | 2,697 | 2,697/0/0 | `msh_Latn` | omnilingual |
| 50 | Antankarana Malagasy | xmv | Madagascar | 43.96 | 2,546 | 1,803/450/293 | `xmv_Latn` | omnilingual |
| 51 | Mauritian Creole | mfe | Mauritius | 43.57 | 15,088 | 13,115/822/1,151 | `mauritian_creole_mfe` | afrispeech |
| 52 | Nyaneka | nyk | Angola | 42.94 | 14,731 | 13,430/858/443 | `nyaneka_nyk` | afrispeech |
| 53 | Pulaar | fuc | Gambia | 42.65 | 2,470 | 2,470/0/0 | `fuc_Latn` | omnilingual |
| 54 | Gun | guw | Benin | 39.98 | 13,340 | 12,157/494/689 | `gun_guw` | afrispeech |
| 55 | Swahili | swa | Tanzania | 38.73 | 13,101 | 11,781/703/617 | `swahili_swa` | afrispeech |
| 56 | Kiluba | lub | DR Congo | 38.31 | 12,949 | 11,553/679/717 | `kiluba_lub` | afrispeech |
| 57 | Igbo | ibo | Nigeria | 38.07 | 12,639 | 11,379/556/704 | `igbo_ibo` | afrispeech |
| 58 | Zulu | zul | South Africa | 36.98 | 12,434 | 11,540/571/323 | `zulu_zul` | afrispeech |
| 59 | Akan | aka | Ghana | 36.37 | 12,752 | 10,107/1,123/1,522 | `aka_asr` | waxal |
| 60 | Changana (Mozambique) | tso | Mozambique | 35.77 | 12,394 | 11,369/478/547 | `changana_mozambique_tso` | afrispeech |
| 61 | Kirundi | run | Burundi | 35.62 | 12,399 | 11,116/550/733 | `kirundi_run` | afrispeech |
| 62 | Kinyarwanda | kin | Rwanda | 35.57 | 12,012 | 10,517/606/889 | `kinyarwanda_kin` | afrispeech |
| 63 | Tsonga | tso | South Africa | 35.25 | 11,838 | 10,710/682/446 | `tsonga_tso` | afrispeech |
| 64 | Ga | gaa | Ghana | 35.12 | 11,939 | 10,703/632/604 | `ga_gaa` | afrispeech |
| 65 | Bara Malagasy | bhr | Madagascar | 33.70 | 1,952 | 1,952/0/0 | `bhr_Latn` | omnilingual |
| 66 | Amharic | amh | Ethiopia | 33.59 | 11,415 | 10,340/603/472 | `amharic_amh` | afrispeech |
| 67 | Fon | fon | Benin | 33.19 | 10,912 | 10,039/459/414 | `fon_fon` | afrispeech |
| 68 | Moroccan Arabic (Arab script) | ary | Algeria | 33.19 | 1,922 | 1,488/0/434 | `ary_Arab` | omnilingual |
| 69 | Xhosa | xho | South Africa | 33.15 | 11,485 | 10,073/658/754 | `xhosa_xho` | afrispeech |
| 70 | Otetela | tll | DR Congo | 31.53 | 11,301 | 10,316/463/522 | `otetela_tll` | afrispeech |
| 71 | Tsimihety Malagasy | xmw | Madagascar | 31.08 | 1,800 | 1,800/0/0 | `xmw_Latn` | omnilingual |
| 72 | Tanosy Malagasy | txy | Madagascar | 30.99 | 1,795 | 898/448/449 | `txy_Latn` | omnilingual |
| 73 | Lingala | lin | DR Congo | 30.71 | 10,080 | 9,076/414/590 | `lingala_lin` | afrispeech |
| 74 | Ewe | ewe | Ghana | 29.41 | 9,760 | 8,633/611/516 | `ewe_ewe` | afrispeech |
| 75 | Chitumbuka | tum | Malawi | 29.31 | 10,365 | 9,303/623/439 | `chitumbuka_tum` | afrispeech |
| 76 | Ronga | rng | Mozambique | 29.11 | 10,285 | 9,138/642/505 | `ronga_rng` | afrispeech |
| 77 | Kamba | kam | Kenya | 28.86 | 9,567 | 8,603/348/616 | `kamba_kam` | afrispeech |
| 78 | Kikuyu | kik | Kenya | 28.82 | 9,640 | 8,808/413/419 | `kikuyu_kik` | afrispeech |
| 79 | Hausa | hau | Nigeria | 28.37 | 9,367 | 8,364/509/494 | `hausa_hau` | afrispeech |
| 80 | Macua | vmw | Mozambique | 28.02 | 10,144 | 8,966/682/496 | `macua_vmw` | afrispeech |
| 81 | Kongo | kon | DR Congo | 27.82 | 9,980 | 8,828/474/678 | `kongo_kon` | afrispeech |
| 82 | Matengo | mgv | Tanzania | 27.32 | 7,792 | 7,402/0/390 | `Matengo` | open-bible |
| 83 | Isoko | iso | Nigeria | 26.84 | 9,474 | 8,544/447/483 | `isoko_iso` | afrispeech |
| 84 | Krio | kri | Sierra Leone | 26.16 | 9,230 | 8,293/553/384 | `krio_kri` | afrispeech |
| 85 | Edo | bin | Nigeria | 26.04 | 9,028 | 8,323/346/359 | `edo_bin` | afrispeech |
| 86 | Bomu | bmq | Burkina Faso | 25.90 | 1,500 | 1,500/0/0 | `bmq_Latn` | omnilingual |
| 87 | Maasina Fulfulde | ffm | Burkina Faso | 25.90 | 1,500 | 1,500/0/0 | `ffm_Latn` | omnilingual |
| 88 | Sakalava Malagasy | skg | Madagascar | 25.88 | 1,499 | 1,499/0/0 | `skg_Latn` | omnilingual |
| 89 | Algerian Arabic (Arab script) | arq | Algeria | 25.71 | 1,489 | 1,489/0/0 | `arq_Arab` | omnilingual |
| 90 | Urhobo | urh | Nigeria | 25.67 | 8,944 | 7,946/447/551 | `urhobo_urh` | afrispeech |
| 91 | Oromo | orm | Ethiopia | 25.66 | 8,933 | 7,778/557/598 | `oromo_orm` | afrispeech |
| 92 | Central-Eastern Niger Fulfulde | fuq | Mali | 25.48 | 1,476 | 600/431/445 | `fuq_Latn` | omnilingual |
| 93 | Saidi Arabic (Arab script) | aec | Egypt | 25.35 | 1,468 | 1,168/150/150 | `aec_Arab` | omnilingual |
| 94 | Masaaba | myx | Uganda | 24.46 | 8,574 | 6,867/849/858 | `mas_asr` | waxal |
| 95 | Esan | ish | Nigeria | 24.19 | 8,499 | 7,356/708/435 | `esan_ish` | afrispeech |
| 96 | Nyankole | nyn | Uganda | 24.17 | 8,472 | 6,783/831/858 | `nyn_asr` | waxal |
| 97 | Fulah | ful | Nigeria | 23.91 | 3,878 | 3,100/380/398 | `ful_tts` | waxal |
| 98 | Frafra | gur | Ghana | 23.82 | 7,960 | 7,140/398/422 | `frafra_gur` | afrispeech |
| 99 | Sesotho (South Africa) | sot | South Africa | 23.32 | 8,149 | 7,583/332/234 | `sesotho_south_africa_sot` | afrispeech |
| 100 | Sena | seh | Mozambique | 22.84 | 7,849 | 7,019/341/489 | `sena_seh` | afrispeech |
| 101 | Sesotho (Lesotho) | sot | South Africa | 22.76 | 7,944 | 7,052/390/502 | `sesotho_lesotho_sot` | afrispeech |
| 102 | Lenje | leh | Zambia | 22.52 | 8,099 | 7,198/412/489 | `lenje_leh` | afrispeech |
| 103 | Soga | xog | Uganda | 22.47 | 7,877 | 6,219/841/817 | `sog_asr` | waxal |
| 104 | Liberian English | lir | Liberia | 22.06 | 7,733 | 7,112/335/286 | `liberian_english_lir` | afrispeech |
| 105 | Fante | fat | Ghana | 22.04 | 7,631 | 6,942/277/412 | `fante_fat` | afrispeech |
| 106 | Dagaare | dga | Ghana | 21.53 | 7,477 | 6,575/375/527 | `dagaare_dga` | afrispeech |
| 107 | Nzema | nzi | Ghana | 21.14 | 7,319 | 6,413/532/374 | `nzema_nzi` | afrispeech |
| 108 | Pidgin (West Africa) | wes | Cameroon | 21.09 | 7,405 | 6,785/227/393 | `pidgin_west_africa_wes` | afrispeech |
| 109 | Kisi | kss | Liberia | 20.99 | 7,739 | 6,967/377/395 | `kisi_kss` | afrispeech |
| 110 | Borgu Fulfulde | fue | Benin | 20.67 | 1,197 | 1,197/0/0 | `fue_Latn` | omnilingual |
| 111 | Fanti | fat | Ghana | 20.67 | 1,197 | 1,197/0/0 | `fat_Latn` | omnilingual |
| 112 | Moore | mos | Burkina Faso | 20.51 | 6,785 | 6,293/196/296 | `moore_mos` | afrispeech |
| 113 | Ahanta | aha | Ghana | 20.30 | 6,908 | 6,038/441/429 | `ahanta_aha` | afrispeech |
| 114 | Gusilay | gsl | Senegal | 20.25 | 1,173 | 775/238/160 | `gsl_Latn` | omnilingual |
| 115 | Rombo | rof | Kenya | 20.20 | 1,170 | 653/241/276 | `rof_Latn` | omnilingual |
| 116 | Douala | dua | Cameroon | 20.12 | 6,521 | 5,832/145/544 | `douala_dua` | afrispeech |
| 117 | Luo | luo | Kenya | 19.66 | 6,829 | 5,959/367/503 | `luo_luo` | afrispeech |
| 118 | Sepedi | nso | South Africa | 19.59 | 7,016 | 6,435/396/185 | `sepedi_nso` | afrispeech |
| 119 | Bissau Guinean Creole | pov | Guinea-Bissau | 19.35 | 6,573 | 5,899/375/299 | `bissau_guinean_creole_pov` | afrispeech |
| 120 | Ganda | lug | Uganda | 19.27 | 6,757 | 5,455/664/638 | `lug_asr` | waxal |
| 121 | Sudanese Arabic (Arab script) | apd | Central African Republic | 18.94 | 1,097 | 816/156/125 | `apd_Arab` | omnilingual |
| 122 | Nobiin | fia | Egypt | 18.87 | 1,093 | 668/262/163 | `fia_Latn` | omnilingual |
| 123 | Swahili (Congo) | swc | DR Congo | 18.67 | 6,493 | 5,670/503/320 | `swahili_congo_swc` | afrispeech |
| 124 | Western Maninkakan | mlq | Gambia | 18.56 | 1,075 | 621/200/254 | `mlq_Latn` | omnilingual |
| 125 | Judeo-Moroccan Arabic (Hebr script) | aju | Morocco | 17.53 | 1,015 | 1,015/0/0 | `aju_Hebr` | omnilingual |
| 126 | Sehwi | sfw | Ghana | 17.48 | 6,024 | 5,488/199/337 | `sehwi_sfw` | afrispeech |
| 127 | Runyankore | nyn | Uganda | 17.10 | 5,605 | 4,736/432/437 | `runyankore_nyn` | afrispeech |
| 128 | Boulou | bum | Cameroon | 16.93 | 5,692 | 5,153/231/308 | `boulou_bum` | afrispeech |
| 129 | Kwanyama | kua | Namibia | 16.66 | 6,107 | 5,467/239/401 | `kwanyama_kua` | afrispeech |
| 130 | Tooro | ttj | DR Congo | 16.54 | 958 | 674/110/174 | `ttj_Latn` | omnilingual |
| 131 | Yoruba | yor | Nigeria | 16.10 | 2,611 | 2,233/201/177 | `yor_tts` | waxal |
| 132 | Konzo | koo | DR Congo | 15.94 | 923 | 654/146/123 | `koo_Latn` | omnilingual |
| 133 | Jula | dyu | Burkina Faso | 15.85 | 5,486 | 5,019/258/209 | `jula_dyu` | afrispeech |
| 134 | Western Niger Fulfulde | fuh | Burkina Faso | 15.64 | 906 | 906/0/0 | `fuh_Latn` | omnilingual |
| 135 | Tandroy-Mahafaly Malagasy | tdx | Madagascar | 15.54 | 900 | 450/150/300 | `tdx_Latn` | omnilingual |
| 136 | Tshwa | tsc | Mozambique | 14.99 | 5,221 | 4,651/209/361 | `tshwa_tsc` | afrispeech |
| 137 | Acoli | ach | Uganda | 14.70 | 5,155 | 4,108/519/528 | `ach_asr` | waxal |
| 138 | Mandjak | mfv | Gambia | 14.28 | 827 | 488/179/160 | `mfv_Latn` | omnilingual |
| 139 | Yoruba | yor | Nigeria | 14.19 | 4,960 | 4,504/216/240 | `yoruba_yor` | afrispeech |
| 140 | Rangi | lag | Tanzania | 14.16 | 820 | 477/180/163 | `lag_Latn` | omnilingual |
| 141 | Balanta-Ganja | bjt | Guinea-Bissau | 13.95 | 808 | 576/109/123 | `bjt_Latn` | omnilingual |
| 142 | Chiga | cgg | DR Congo | 13.90 | 805 | 559/126/120 | `cgg_Latn` | omnilingual |
| 143 | Setswana | tsn | South Africa | 13.84 | 4,828 | 4,232/344/252 | `setswana_tsn` | afrispeech |
| 144 | Kikongo ya Leta | ktu | DR Congo | 13.81 | 4,754 | 4,232/262/260 | `kikongo_ya_leta_ktu` | afrispeech |
| 145 | Liberian English | lir | Liberia | 13.76 | 797 | 650/71/76 | `lir_Latn` | omnilingual |
| 146 | Gambian Wolof | wof | Gambia | 13.21 | 765 | 548/116/101 | `wof_Latn` | omnilingual |
| 147 | Chichewa | nya | Malawi | 12.81 | 3,884 | 3,407/231/246 | `chichewa_nya` | afrispeech |
| 148 | Akebu | keu | Togo | 12.67 | 734 | 494/91/149 | `keu_Latn` | omnilingual |
| 149 | Sango | sag | Central African Republic | 12.60 | 4,193 | 3,645/261/287 | `sango_sag` | afrispeech |
| 150 | Acoli | ach | Uganda | 12.52 | 2,031 | 1,621/218/192 | `ach_tts` | waxal |
| 151 | Kikuyu | kik | Kenya | 12.49 | 2,026 | 1,602/210/214 | `kik_tts` | waxal |
| 152 | Ganda | lug | Uganda | 12.48 | 2,024 | 1,608/211/205 | `lug_tts` | waxal |
| 153 | Mashi | shr | DR Congo | 12.41 | 4,263 | 3,855/273/135 | `mashi_shr` | afrispeech |
| 154 | Nigerian Pidgin | pcm | Nigeria | 12.29 | 1,993 | 1,590/199/204 | `pcm_tts` | waxal |
| 155 | Luo (Kenya and Tanzania) | luo | Kenya | 12.26 | 1,989 | 1,552/228/209 | `luo_tts` | waxal |
| 156 | Nyankole | nyn | Uganda | 12.25 | 1,987 | 1,597/199/191 | `nyn_tts` | waxal |
| 157 | Seychelles Creole | crs | Seychelles | 12.16 | 4,269 | 3,983/72/214 | `seychelles_creole_crs` | afrispeech |
| 158 | Hausa | hau | Nigeria | 12.15 | 1,971 | 1,572/202/197 | `hau_tts` | waxal |
| 159 | Bayot | bda | Gambia | 11.83 | 685 | 450/158/77 | `bda_Latn` | omnilingual |
| 160 | Igbo | ibo | Nigeria | 11.78 | 1,911 | 1,552/159/200 | `ibo_tts` | waxal |
| 161 | Fang | fan | Equatorial Guinea | 11.68 | 3,969 | 3,630/184/155 | `fang_fan` | afrispeech |
| 162 | Tugen | tuy | Kenya | 11.52 | 667 | 407/104/156 | `tuy_Latn` | omnilingual |
| 163 | Fipa | fip | Malawi | 11.46 | 664 | 410/106/148 | `fip_Latn` | omnilingual |
| 164 | Gweno | gwe | Kenya | 11.34 | 657 | 401/123/133 | `gwe_Latn` | omnilingual |
| 165 | Luganda | lug | Uganda | 11.25 | 3,617 | 3,237/75/305 | `luganda_lug` | afrispeech |
| 166 | Kituba | ktu | DR Congo | 11.24 | 3,935 | 3,639/195/101 | `kituba_ktu` | afrispeech |
| 167 | Tshiluba | lua | DR Congo | 11.24 | 3,532 | 3,234/158/140 | `tshiluba_lua` | afrispeech |
| 168 | Keiyo | eyo | Kenya | 11.17 | 647 | 398/109/140 | `eyo_Latn` | omnilingual |
| 169 | Farefare | gur | Burkina Faso | 11.08 | 642 | 476/66/100 | `gur_Latn` | omnilingual |
| 170 | Swahili (macrolanguage) | swa | Kenya | 10.96 | 1,778 | 1,387/192/199 | `swa_tts` | waxal |
| 171 | Chopi | cce | Mozambique | 10.80 | 3,789 | 3,414/144/231 | `chopi_cce` | afrispeech |
| 172 | Warji | wji | Nigeria | 10.62 | 615 | 443/98/74 | `wji_Latn` | omnilingual |
| 173 | Cibemba | bem | Zambia | 10.56 | 3,823 | 3,333/170/320 | `cibemba_bem` | afrispeech |
| 174 | Bago-Kusuntu | bqg | Togo | 10.52 | 609 | 352/99/158 | `bqg_Latn` | omnilingual |
| 175 | Anaang | anw | Nigeria | 10.41 | 603 | 369/115/119 | `anw_Latn` | omnilingual |
| 176 | Tahaggart Tamahaq (Tfng script) | thv | Algeria | 10.29 | 596 | 298/149/149 | `thv_Tfng` | omnilingual |
| 177 | Ngangela | nba | Angola | 10.28 | 3,638 | 3,348/166/124 | `ngangela_nba` | afrispeech |
| 178 | Gola | gol | Liberia | 10.24 | 593 | 347/120/126 | `gol_Latn` | omnilingual |
| 179 | Ndau (Western) | ndc | Mozambique | 10.22 | 3,621 | 3,235/239/147 | `ndau_western_ndc` | afrispeech |
| 180 | Gbari | gby | Nigeria | 10.01 | 580 | 409/97/74 | `gby_Latn` | omnilingual |
| 181 | Phimbi | phm | Mozambique | 10.00 | 3,561 | 3,172/121/268 | `phimbi_phm` | afrispeech |
| 182 | Gbagyi | gbr | Nigeria | 9.82 | 569 | 396/94/79 | `gbr_Latn` | omnilingual |
| 183 | Dũya | ldb | Nigeria | 9.76 | 565 | 401/74/90 | `ldb_Latn` | omnilingual |
| 184 | Dera (Nigeria) | kna | Nigeria | 9.69 | 561 | 376/77/108 | `kna_Latn` | omnilingual |
| 185 | Igo | ahl | Ghana | 9.58 | 555 | 376/87/92 | `ahl_Latn` | omnilingual |
| 186 | Mansoanka | msw | Gambia | 9.58 | 555 | 379/88/88 | `msw_Latn` | omnilingual |
| 187 | Samba Leko | ndi | Cameroon | 9.58 | 555 | 349/134/72 | `ndi_Latn` | omnilingual |
| 188 | Soninke | snk | Gambia | 9.57 | 554 | 385/91/78 | `snk_Latn` | omnilingual |
| 189 | Pero | pip | Nigeria | 9.50 | 550 | 322/124/104 | `pip_Latn` | omnilingual |
| 190 | Bondei | bou | Tanzania | 9.43 | 546 | 373/103/70 | `bou_Latn` | omnilingual |
| 191 | Guduf-Gava | gdf | Cameroon | 9.43 | 546 | 376/102/68 | `gdf_Latn` | omnilingual |
| 192 | Swati | ssw | South Africa | 9.42 | 3,236 | 2,767/170/299 | `swati_ssw` | afrispeech |
| 193 | Cakfem-Mushere | cky | Nigeria | 9.41 | 545 | 307/123/115 | `cky_Latn` | omnilingual |
| 194 | Ewe | ewe | Ghana | 9.36 | 1,519 | 1,215/152/152 | `ewe_tts` | waxal |
| 195 | Burak | bys | Nigeria | 9.31 | 539 | 360/103/76 | `bys_Latn` | omnilingual |
| 196 | Idoma | idu | Nigeria | 9.24 | 535 | 348/115/72 | `idu_Latn` | omnilingual |
| 197 | Kalabari | ijn | Nigeria | 9.24 | 535 | 341/94/100 | `ijn_Latn` | omnilingual |
| 198 | Vai | vai | Liberia | 9.20 | 533 | 299/87/147 | `vai_Latn` | omnilingual |
| 199 | Isekiri | its | Nigeria | 9.19 | 532 | 369/80/83 | `its_Latn` | omnilingual |
| 200 | Ngizim | ngi | Nigeria | 9.13 | 529 | 352/104/73 | `ngi_Latn` | omnilingual |
| 201 | Eastern Krahn | kqo | Côte d'Ivoire | 9.12 | 528 | 280/106/142 | `kqo_Latn` | omnilingual |
| 202 | Eloyi | afo | Nigeria | 9.10 | 527 | 357/70/100 | `afo_Latn` | omnilingual |
| 203 | Naba | mne | Chad | 9.08 | 526 | 251/176/99 | `mne_Latn` | omnilingual |
| 204 | Alago | ala | Nigeria | 9.05 | 524 | 341/86/97 | `ala_Latn` | omnilingual |
| 205 | Lijili | mgi | Nigeria | 9.00 | 521 | 353/70/98 | `mgi_Latn` | omnilingual |
| 206 | Ubaghara | byc | Nigeria | 9.00 | 521 | 349/86/86 | `byc_Latn` | omnilingual |
| 207 | Kinga | zga | Malawi | 8.98 | 520 | 520/0/0 | `zga_Latn` | omnilingual |
| 208 | Toupouri | tui | Cameroon | 8.96 | 3,070 | 2,588/292/190 | `toupouri_tui` | afrispeech |
| 209 | Abron | abr | Côte d'Ivoire | 8.94 | 518 | 302/87/129 | `abr_Latn` | omnilingual |
| 210 | Eleme | elm | Nigeria | 8.91 | 516 | 350/90/76 | `elm_Latn` | omnilingual |
| 211 | Mbe | mfo | Nigeria | 8.87 | 514 | 341/80/93 | `mfo_Latn` | omnilingual |
| 212 | Khana | ogo | Nigeria | 8.79 | 509 | 332/77/100 | `ogo_Latn` | omnilingual |
| 213 | Degema | deg | Nigeria | 8.77 | 508 | 339/90/79 | `deg_Latn` | omnilingual |
| 214 | Ejagham | etu | Cameroon | 8.77 | 508 | 325/95/88 | `etu_Latn` | omnilingual |
| 215 | Ikwere | ikw | Nigeria | 8.70 | 504 | 318/91/95 | `ikw_Latn` | omnilingual |
| 216 | Goemai | ank | Nigeria | 8.68 | 503 | 325/107/71 | `ank_Latn` | omnilingual |
| 217 | Isoko | iso | Nigeria | 8.68 | 503 | 325/79/99 | `iso_Latn` | omnilingual |
| 218 | Bura-Pabir | bwr | Nigeria | 8.67 | 502 | 352/82/68 | `bwr_Latn` | omnilingual |
| 219 | Venda | ven | South Africa | 8.65 | 2,990 | 2,721/135/134 | `venda_ven` | afrispeech |
| 220 | Basa (Nigeria) | bzw | Nigeria | 8.63 | 500 | 336/72/92 | `bzw_Latn` | omnilingual |
| 221 | Yace | ekr | Nigeria | 8.63 | 500 | 351/74/75 | `ekr_Latn` | omnilingual |
| 222 | Ito | itw | Nigeria | 8.58 | 497 | 341/79/77 | `itw_Latn` | omnilingual |
| 223 | Miya | mkf | Nigeria | 8.58 | 497 | 342/67/88 | `mkf_Latn` | omnilingual |
| 224 | Nigerian Fulfulde | fuv | Cameroon | 8.56 | 496 | 347/74/75 | `fuv_Latn` | omnilingual |
| 225 | Abua | abn | Nigeria | 8.55 | 495 | 317/86/92 | `abn_Latn` | omnilingual |
| 226 | Ngamo | nbh | Nigeria | 8.53 | 494 | 311/73/110 | `nbh_Latn` | omnilingual |
| 227 | Kuanyama | kua | Angola | 8.48 | 491 | 312/95/84 | `kua_Latn` | omnilingual |
| 228 | Kwambi | kwm | Namibia | 8.48 | 491 | 328/75/88 | `kwm_Latn` | omnilingual |
| 229 | Waja | wja | Nigeria | 8.48 | 491 | 345/73/73 | `wja_Latn` | omnilingual |
| 230 | Geji | gyz | Nigeria | 8.46 | 490 | 315/92/83 | `gyz_Latn` | omnilingual |
| 231 | Bade | bde | Nigeria | 8.44 | 489 | 314/101/74 | `bde_Latn` | omnilingual |
| 232 | Tsotso | lto | Kenya | 8.44 | 489 | 316/98/75 | `lto_Latn` | omnilingual |
| 233 | Logooli | rag | Kenya | 8.43 | 488 | 306/72/110 | `rag_Latn` | omnilingual |
| 234 | Dijim-Bwilim | cfa | Nigeria | 8.41 | 487 | 334/69/84 | `cfa_Latn` | omnilingual |
| 235 | Wapan | juk | Nigeria | 8.41 | 487 | 307/75/105 | `juk_Latn` | omnilingual |
| 236 | Boghom | bux | Nigeria | 8.39 | 486 | 273/118/95 | `bux_Latn` | omnilingual |
| 237 | Cibak | ckl | Nigeria | 8.39 | 486 | 299/78/109 | `ckl_Latn` | omnilingual |
| 238 | Kohumono | bcs | Nigeria | 8.39 | 486 | 347/70/69 | `bcs_Latn` | omnilingual |
| 239 | Mom Jango | ver | Cameroon | 8.39 | 486 | 323/68/95 | `ver_Latn` | omnilingual |
| 240 | Ashe | ahs | Nigeria | 8.36 | 484 | 329/84/71 | `ahs_Latn` | omnilingual |
| 241 | Nyungwe | nyu | Mozambique | 8.35 | 2,976 | 2,744/63/169 | `nyungwe_nyu` | afrispeech |
| 242 | Wanga | lwg | Kenya | 8.34 | 483 | 297/76/110 | `lwg_Latn` | omnilingual |
| 243 | Kimbundu | kmb | Angola | 8.32 | 2,894 | 2,468/249/177 | `kimbundu_kmb` | afrispeech |
| 244 | Mafa | maf | Cameroon | 8.32 | 482 | 325/88/69 | `maf_Latn` | omnilingual |
| 245 | Odual | odu | Nigeria | 8.32 | 482 | 307/90/85 | `odu_Latn` | omnilingual |
| 246 | Esan | ish | Nigeria | 8.31 | 481 | 308/98/75 | `ish_Latn` | omnilingual |
| 247 | Gusii | guz | Kenya | 8.29 | 480 | 287/117/76 | `guz_Latn` | omnilingual |
| 248 | Turkana | tuv | Ethiopia | 8.27 | 479 | 285/106/88 | `tuv_Latn` | omnilingual |
| 249 | Yekhee | ets | Nigeria | 8.25 | 478 | 320/79/79 | `ets_Latn` | omnilingual |
| 250 | Cen | cen | Nigeria | 8.24 | 477 | 313/90/74 | `cen_Latn` | omnilingual |
| 251 | Hwana | hwo | Nigeria | 8.24 | 477 | 327/68/82 | `hwo_Latn` | omnilingual |
| 252 | Koma | kmy | Cameroon | 8.24 | 477 | 326/65/86 | `kmy_Latn` | omnilingual |
| 253 | Ikposo | kpo | Ghana | 8.22 | 476 | 293/90/93 | `kpo_Latn` | omnilingual |
| 254 | Tangale | tan | Nigeria | 8.22 | 476 | 323/69/84 | `tan_Latn` | omnilingual |
| 255 | Karekare | kai | Nigeria | 8.18 | 474 | 331/69/74 | `kai_Latn` | omnilingual |
| 256 | Lamang | hia | Nigeria | 8.18 | 474 | 324/85/65 | `hia_Latn` | omnilingual |
| 257 | Cross River Mbembe | mfn | Nigeria | 8.17 | 473 | 284/89/100 | `mfn_Latn` | omnilingual |
| 258 | Idakho-Isukha-Tiriki | ida | Kenya | 8.17 | 473 | 305/100/68 | `ida_Latn` | omnilingual |
| 259 | Kinande | nnb | DR Congo | 8.15 | 2,871 | 2,637/44/190 | `kinande_nnb` | afrispeech |
| 260 | Huba | hbb | Nigeria | 8.12 | 470 | 288/63/119 | `hbb_Latn` | omnilingual |
| 261 | Marghi Central | mrt | Nigeria | 8.12 | 470 | 297/80/93 | `mrt_Latn` | omnilingual |
| 262 | Waci Gbe | wci | Benin | 8.12 | 470 | 308/95/67 | `wci_Latn` | omnilingual |
| 263 | Kamo | kcq | Nigeria | 8.10 | 469 | 321/66/82 | `kcq_Latn` | omnilingual |
| 264 | Nyankpa | yes | Nigeria | 8.10 | 469 | 310/68/91 | `yes_Latn` | omnilingual |
| 265 | Nsenga (Mozambique) | nse | Mozambique | 8.09 | 2,865 | 2,385/313/167 | `nsenga_mozambique_nse` | afrispeech |
| 266 | Baoule | bci | Côte d'Ivoire | 8.08 | 2,657 | 2,307/125/225 | `baoule_bci` | afrispeech |
| 267 | Bokyi | bky | Cameroon | 8.08 | 468 | 297/93/78 | `bky_Latn` | omnilingual |
| 268 | Jiba | juo | Nigeria | 8.06 | 467 | 321/68/78 | `juo_Latn` | omnilingual |
| 269 | Piya-Kwonci | piy | Nigeria | 8.06 | 467 | 324/76/67 | `piy_Latn` | omnilingual |
| 270 | Dadiya | dbd | Nigeria | 8.05 | 466 | 289/70/107 | `dbd_Latn` | omnilingual |
| 271 | Izon | ijc | Nigeria | 8.05 | 466 | 290/80/96 | `ijc_Latn` | omnilingual |
| 272 | Pökoot | pko | Kenya | 8.03 | 465 | 298/70/97 | `pko_Latn` | omnilingual |
| 273 | Lala-Roba | lla | Nigeria | 7.99 | 463 | 301/91/71 | `lla_Latn` | omnilingual |
| 274 | Gokana | gkn | Nigeria | 7.98 | 2,808 | 2,567/179/62 | `gokana_gkn` | afrispeech |
| 275 | Jju | kaj | Nigeria | 7.96 | 461 | 272/91/98 | `kaj_Latn` | omnilingual |
| 276 | Kabras | lkb | Kenya | 7.96 | 461 | 284/72/105 | `lkb_Latn` | omnilingual |
| 277 | Longuda | lnu | Nigeria | 7.96 | 461 | 299/63/99 | `lnu_Latn` | omnilingual |
| 278 | Nzanyi | nja | Cameroon | 7.94 | 460 | 302/87/71 | `nja_Latn` | omnilingual |
| 279 | Tarok | yer | Nigeria | 7.94 | 460 | 298/70/92 | `yer_Latn` | omnilingual |
| 280 | Buduma | bdm | Cameroon | 7.93 | 459 | 301/90/68 | `bdm_Latn` | omnilingual |
| 281 | Ngas | anc | Nigeria | 7.91 | 458 | 314/65/79 | `anc_Latn` | omnilingual |
| 282 | Kushi | kuh | Nigeria | 7.89 | 457 | 308/77/72 | `kuh_Latn` | omnilingual |
| 283 | Marghi South | mfm | Nigeria | 7.89 | 457 | 315/78/64 | `mfm_Latn` | omnilingual |
| 284 | Ndonga | ndo | Angola | 7.89 | 457 | 285/68/104 | `ndo_Latn` | omnilingual |
| 285 | Samba Daka | ccg | Cameroon | 7.89 | 457 | 292/95/70 | `ccg_Latn` | omnilingual |
| 286 | Saya | say | Nigeria | 7.89 | 457 | 260/103/94 | `say_Latn` | omnilingual |
| 287 | Eggon | ego | Nigeria | 7.87 | 456 | 305/68/83 | `ego_Latn` | omnilingual |
| 288 | Embu | ebu | Kenya | 7.87 | 456 | 272/71/113 | `ebu_Latn` | omnilingual |
| 289 | Glavda | glw | Cameroon | 7.87 | 456 | 322/65/69 | `glw_Latn` | omnilingual |
| 290 | Kanembu | kbl | Niger | 7.87 | 456 | 306/79/71 | `kbl_Latn` | omnilingual |
| 291 | Kirya-Konzəl | fkk | Nigeria | 7.87 | 456 | 290/88/78 | `fkk_Latn` | omnilingual |
| 292 | Kulung (Nigeria) | bbu | Nigeria | 7.84 | 454 | 281/67/106 | `bbu_Latn` | omnilingual |
| 293 | Meru | mer | Kenya | 7.80 | 452 | 279/105/68 | `mer_Latn` | omnilingual |
| 294 | Maba (Chad) | mde | Chad | 7.68 | 445 | 445/0/0 | `mde_Latn` | omnilingual |
| 295 | Tula | tul | Nigeria | 7.68 | 445 | 263/112/70 | `tul_Latn` | omnilingual |
| 296 | Afade | aal | Cameroon | 7.67 | 444 | 310/64/70 | `aal_Latn` | omnilingual |
| 297 | Bacama | bcy | Nigeria | 7.65 | 443 | 278/99/66 | `bcy_Latn` | omnilingual |
| 298 | Herero | her | Angola | 7.65 | 443 | 281/95/67 | `her_Latn` | omnilingual |
| 299 | Ndebele (Zimbabwe) | nde | Zimbabwe | 7.62 | 2,552 | 2,308/81/163 | `ndebele_zimbabwe_nde` | afrispeech |
| 300 | Zarma | dje | Burkina Faso | 7.61 | 441 | 316/61/64 | `dje_Latn` | omnilingual |
| 301 | Havu | hav | DR Congo | 7.59 | 2,619 | 2,447/172/0 | `havu_hav` | afrispeech |
| 302 | Bangwinji | bsj | Nigeria | 7.58 | 439 | 286/84/69 | `bsj_Latn` | omnilingual |
| 303 | Dghwede | dgh | Cameroon | 7.58 | 439 | 263/109/67 | `dgh_Latn` | omnilingual |
| 304 | Tedaga | tuq | Libya | 7.56 | 438 | 287/87/64 | `tuq_Latn` | omnilingual |
| 305 | Tera | ttr | Nigeria | 7.53 | 436 | 294/75/67 | `ttr_Latn` | omnilingual |
| 306 | Baoulé | bci | Côte d'Ivoire | 7.50 | 1,216 | 972/122/122 | `bau_tts` | waxal |
| 307 | Ibinda | yom | DR Congo | 7.46 | 2,672 | 2,508/59/105 | `ibinda_yom` | afrispeech |
| 308 | Dazaga | dzg | Libya | 7.39 | 428 | 261/65/102 | `dzg_Latn` | omnilingual |
| 309 | Awak | awo | Nigeria | 7.37 | 427 | 265/71/91 | `awo_Latn` | omnilingual |
| 310 | Fanti | fat | Ghana | 7.22 | 1,171 | 953/117/101 | `fat_tts` | waxal |
| 311 | Liberia Kpelle | xpe | Guinea | 7.22 | 418 | 266/84/68 | `xpe_Latn` | omnilingual |
| 312 | Agwagwune | yay | Nigeria | 6.84 | 396 | 227/87/82 | `yay_Latn` | omnilingual |
| 313 | Dinka | din | South Sudan | 6.84 | 2,496 | 2,178/99/219 | `dinka_din` | afrispeech |
| 314 | Itsekiri | its | Nigeria | 6.78 | 2,387 | 2,387/0/0 | `itsekiri_its` | afrispeech |
| 315 | Kwangali | kwn | Namibia | 6.75 | 2,258 | 2,139/81/38 | `kwangali_kwn` | afrispeech |
| 316 | Twi | twi | Ghana | 6.74 | 1,093 | 872/117/104 | `twi_tts` | waxal |
| 317 | Chitonga | toi | Zambia | 6.61 | 2,487 | 2,220/195/72 | `chitonga_toi` | afrispeech |
| 318 | Bassa (Liberia) | bsq | Liberia | 6.07 | 2,284 | 2,135/12/137 | `bassa_liberia_bsq` | afrispeech |
| 319 | Cinyanja | nya | Malawi | 5.79 | 1,883 | 1,765/42/76 | `cinyanja_nya` | afrispeech |
| 320 | Ndonga | ndo | Namibia | 5.77 | 1,920 | 1,745/63/112 | `ndonga_ndo` | afrispeech |
| 321 | Aja | ajg | Benin | 5.71 | 1,942 | 1,765/105/72 | `aja_ajg` | afrispeech |
| 322 | Kpelle | xpe | Liberia | 5.70 | 1,933 | 1,743/112/78 | `kpelle_xpe` | afrispeech |
| 323 | Réunion Creole | rcf | Réunion | 5.25 | 1,803 | 1,718/23/62 | `r_union_creole_rcf` | afrispeech |
| 324 | Eastern Egyptian Bedawi Arabic (Arab script) | avl | Egypt | 5.18 | 300 | 300/0/0 | `avl_Arab` | omnilingual |
| 325 | Algerian Saharan Arabic (Arab script) | aao | Algeria | 5.15 | 298 | 298/0/0 | `aao_Arab` | omnilingual |
| 326 | Ndebele | nbl | South Africa | 5.15 | 1,985 | 1,819/134/32 | `ndebele_nbl` | afrispeech |
| 327 | Chadian Arabic | shu | Cameroon | 5.13 | 297 | 297/0/0 | `shu_Latn` | omnilingual |
| 328 | Abbey | aba | Côte d'Ivoire | 5.03 | 1,776 | 1,574/166/36 | `abbey_aba` | afrispeech |
| 329 | Yombe | yom | DR Congo | 4.88 | 1,634 | 1,390/149/95 | `yombe_yom` | afrispeech |
| 330 | Kikongo | kwy | Angola | 4.87 | 1,769 | 1,623/103/43 | `kikongo_kwy` | afrispeech |
| 331 | Umbundu | umb | Angola | 4.70 | 1,630 | 1,340/227/63 | `umbundu_umb` | afrispeech |
| 332 | Chiyao | yao | Mozambique | 4.61 | 1,647 | 1,423/103/121 | `chiyao_yao` | afrispeech |
| 333 | Loma | lom | Liberia | 4.52 | 1,540 | 1,350/41/149 | `loma_lom` | afrispeech |
| 334 | Wolaita | wal | Ethiopia | 4.28 | 1,476 | 1,250/133/93 | `wolaita_wal` | afrispeech |
| 335 | Chitonga (Malawi) | tog | Malawi | 4.16 | 1,480 | 1,363/65/52 | `chitonga_malawi_tog` | afrispeech |
| 336 | Tiv | tiv | Nigeria | 4.03 | 1,374 | 1,220/119/35 | `tiv_tiv` | afrispeech |
| 337 | Lari | ldi | Congo | 3.85 | 1,359 | 1,284/42/33 | `lari_ldi` | afrispeech |
| 338 | Meru | mer | Kenya | 3.83 | 1,277 | 1,139/138/0 | `meru_mer` | afrispeech |
| 339 | Ewondo | ewo | Cameroon | 3.71 | 1,305 | 1,134/32/139 | `ewondo_ewo` | afrispeech |
| 340 | Kabyle | kab | Algeria | 3.65 | 1,188 | 1,068/64/56 | `kabyle_kab` | afrispeech |
| 341 | Khana | ogo | Nigeria | 3.62 | 1,256 | 921/139/196 | `khana_ogo` | afrispeech |
| 342 | Gitonga | toh | Mozambique | 3.60 | 1,319 | 1,228/4/87 | `gitonga_toh` | afrispeech |
| 343 | Tewe | twx | Mozambique | 3.39 | 1,251 | 1,076/148/27 | `tewe_twx` | afrispeech |
| 344 | Dangme | ada | Ghana | 3.37 | 1,177 | 941/154/82 | `dangme_ada` | afrispeech |
| 345 | Ndau | ndc | Mozambique | 3.34 | 1,231 | 1,147/68/16 | `ndau_ndc` | afrispeech |
| 346 | Guéré | gxx | Côte d'Ivoire | 3.01 | 1,057 | 953/67/37 | `gu_r_gxx` | afrispeech |
| 347 | Wolof | wol | Senegal | 2.50 | 846 | 803/10/33 | `wolof_wol` | afrispeech |
| 348 | Damara | naq | Namibia | 2.46 | 789 | 698/62/29 | `damara_naq` | afrispeech |
| 349 | Swahili (Katanga) | swc | DR Congo | 2.40 | 866 | 798/32/36 | `swahili_katanga_swc` | afrispeech |
| 350 | Yacouba | daf | Côte d'Ivoire | 2.26 | 787 | 656/0/131 | `yacouba_daf` | afrispeech |
| 351 | Manyawa | mny | Mozambique | 1.94 | 697 | 697/0/0 | `manyawa_mny` | afrispeech |
| 352 | Makhuwa-Marrevone | xmc | Mozambique | 1.87 | 704 | 635/41/28 | `makhuwa_marrevone_xmc` | afrispeech |
| 353 | Makhuwa-Meetto | mgh | Mozambique | 1.83 | 671 | 552/85/34 | `makhuwa_meetto_mgh` | afrispeech |
| 354 | Cinamwanga | mwn | Zambia | 1.66 | 553 | 500/53/0 | `cinamwanga_mwn` | afrispeech |
| 355 | Chitonga (Zimbabwe) | toi | Zimbabwe | 1.41 | 486 | 452/0/34 | `chitonga_zimbabwe_toi` | afrispeech |
| 356 | Attié | ati | Côte d'Ivoire | 1.40 | 482 | 477/5/0 | `atti_ati` | afrispeech |
| 357 | Lunda | lun | Zambia | 1.11 | 423 | 322/101/0 | `lunda_lun` | afrispeech |
| 358 | Lomwe | ngl | Mozambique | 1.07 | 378 | 373/5/0 | `lomwe_ngl` | afrispeech |
| 359 | Chuabo | chw | Mozambique | 1.03 | 390 | 373/0/17 | `chuabo_chw` | afrispeech |
| 360 | Mambwe-Lungu | mgr | Zambia | 0.93 | 331 | 264/34/33 | `mambwe_lungu_mgr` | afrispeech |
| 361 | Ijaw | ijc | Nigeria | 0.88 | 312 | 292/0/20 | `ijaw_ijc` | afrispeech |
| 362 | Ngbandi (Northern) | ngb | DR Congo | 0.78 | 255 | 222/33/0 | `ngbandi_northern_ngb` | afrispeech |
| 363 | Makhuwa-Shirima | vmk | Mozambique | 0.77 | 277 | 277/0/0 | `makhuwa_shirima_vmk` | afrispeech |
| 364 | Herero | her | Namibia | 0.61 | 192 | 192/0/0 | `herero_her` | afrispeech |
| 365 | Chokwe | cjk | Angola | 0.56 | 177 | 169/0/8 | `chokwe_cjk` | afrispeech |
| 366 | Taabwa | tap | DR Congo | 0.56 | 195 | 163/0/32 | `taabwa_tap` | afrispeech |
| 367 | Kisonge | sop | DR Congo | 0.40 | 141 | 141/0/0 | `kisonge_sop` | afrispeech |
| 368 | Kanyok | kny | DR Congo | 0.28 | 109 | 109/0/0 | `kanyok_kny` | afrispeech |
| 369 | Luvale | lue | Zambia | 0.07 | 20 | 20/0/0 | `luvale_lue` | afrispeech |

## License

CC-BY-4.0
