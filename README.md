# AfriSpeech Selector

**AfriSpeech Selector gives you access to 142 languages · 2267.9 hours · 35
countries of African speech for training TTS and ASR models.**

Select languages by recorded **hours** (strength) — a country-balanced top-N or a
hand-picked set, sized the way you want — and get the **audio + metadata in the
format your training pipeline expects**. You take it from there; the tool doesn't
do any text normalisation or cleaning (that's your framework's job).

- **TTS data-prep** — export WAVs + a manifest in the layout your framework reads:
  **LJSpeech, Piper, VITS, or MeloTTS**.
- **ASR / 🤗 datasets** — export `load_from_disk` / Parquet, or stream straight in
  with no local copy (`stream_dataset(...)`).

Size it however you like: a **total budget** (e.g. `--total-hours 16`), **per
language** (clips or hours), or everything available.

**Source & redistribution:** the audio comes from the AfriSpeech
[`african-speech-public_v1`](https://huggingface.co/datasets/AfriSpeech/african-speech-public_v1)
dataset on the Hugging Face Hub. A *local* working set for your own training is
fine; redistributing copies (e.g. `--push` to a public repo) is **not
recommended**, given the permissions of the public data sources the dataset
draws on.

## Available languages

**142 languages · 2267.9 hours · 35 countries.** Hours is the strength signal
used for ranking. The `--languages` value is the name you pass to pick a language
(e.g. `--languages twi_twi`). List them anytime with `afrispeech-select --list-langs`, or see the full **[language catalog](#language-catalog)** at the bottom.


## Install

```bash
git clone https://github.com/AfriSpeech/afrispeech-selector.git
cd afrispeech-selector
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # gives you the `afrispeech-select` command
```

(Use `python3` — the code uses non-ASCII text and won't run under Python 2.)

## Quickstart — one language, ready to train

Most people just want one language. Pick its name (`afrispeech-select --list-langs`,
or the table below) and run one command. Clips are filtered to a **3–15 s** window
by default, so the result is training-ready.

```bash
# ~5 hours of Twi as an LJSpeech TTS dataset (wavs/ + metadata.csv), 22.05 kHz
afrispeech-select --languages twi_twi --total-hours 5 --out ./twi --format ljspeech

# …for Piper / VITS / MeloTTS instead, just change --format:
afrispeech-select --languages twi_twi --total-hours 5 --out ./twi --format piper

# …for ASR (Hugging Face datasets on disk, 16 kHz):
afrispeech-select --languages twi_twi --total-hours 5 --out ./twi --format disk --target-sr 16000
```

That's it — `./twi` now holds the audio + metadata in the right layout. Want more
or less? Change `--total-hours` (or use `--per-language N` for a clip count).
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
afrispeech-select --languages twi_twi --total-hours 5 --out ./twi --format ljspeech

# Piper (metadata.csv: id|speaker|text)
afrispeech-select --languages twi_twi --total-hours 5 --out ./twi --format piper

# VITS (filelist.txt + speakers.txt) and/or MeloTTS (metadata.list)
afrispeech-select --languages twi_twi --total-hours 5 --out ./twi --format vits,melo
```

Each writes `<out>/wavs/*.wav` plus the manifest. Phonemisation / text cleaning
is left to the framework's own preprocessor.

## Export one language for ASR / 🤗 datasets

```bash
# On-disk dataset (load_from_disk) + a parquet file, resampled to 16 kHz
afrispeech-select --languages twi_twi --total-hours 5 --out ./twi \
    --format disk,parquet --target-sr 16000
```

```python
from datasets import load_from_disk
ds = load_from_disk("twi").train_test_split(test_size=0.1)
```

Or stream it in with **no local copy** (lazy `IterableDataset`):

```python
from afrispeech_selector import stream_dataset
ds = stream_dataset(["twi_twi"], split="train", max_seconds=5 * 3600,
                    target_sampling_rate=16000)   # min/max clip default to 3/15s
for batch in ds.iter(batch_size=8):
    ...
# `afrispeech-select … --recipe` prints this snippet for any selection.
```

No install? `python3 -m afrispeech_selector …` works the same from the repo.

## Use it in a Colab / Kaggle training notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AfriSpeech/afrispeech-selector/blob/main/notebooks/afrispeech_selector.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/AfriSpeech/afrispeech-selector/blob/main/notebooks/afrispeech_selector.ipynb)

The example notebook is meant to be **dropped into your own training notebook**:
its cells install the tool and pull your selected language(s) **straight into the
running session** — stream into ASR training, or write `wavs/` + a manifest for a
TTS framework — so your training script has the data without manual downloads.
It's for *getting data into training*, not for making dataset copies. (On Kaggle,
turn **Internet ON**.) See
[`notebooks/afrispeech_selector.ipynb`](notebooks/afrispeech_selector.ipynb).

## Selecting multiple languages

Everything above works for several languages too — the format/output flags are
identical, you just choose *which* languages. `--total-hours` is split evenly
across them (so each gets its fair share).

**1. Name the languages you want:**

```bash
# Ghanaian languages, 15 h total (5 h each), LJSpeech
afrispeech-select --languages twi_twi,ewe_ewe,ga_gaa --total-hours 15 \
    --out ./gh --format ljspeech
```

**2. Or pick across the dataset by strength / balance** — top-N by hours, with
optional country balancing and pool filters:

```bash
# Top 10 languages, at most 2 per country, 24 h total
afrispeech-select --top 10 --max-per-country 2 --total-hours 24 \
    --out ./multi --format ljspeech

# Narrow the pool first: only languages >= 20 h, only Ghana & Nigeria
afrispeech-select --top 8 --min-hours 20 --countries GH,NG --total-hours 16 \
    --out ./wa --format ljspeech
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
An uncapped "full build" downloads whole shards (the dataset is ~65 GB) and must
be enabled with `--allow-full`.

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
  builder.py    pull subsets → standard schema; stream_dataset (no copy) + apply_schema
  export.py     HF (zip/parquet/manifest/push) + TTS (ljspeech/piper/vits/melo)
app.py          optional selection UI (emits the CLI command)
data/catalog.tsv  strength table (hours, clips, splits per subset)
scripts/        refresh_catalog.py, clean_source_dataset.py
tests/          selection, builder, CLI tests
```

## Language catalog

All **142 languages**, sorted by hours (strength). The `--languages` value is what you pass on the CLI.

| # | Language | ISO | Country | Hours | Clips | Train/Val/Test | `--languages` value |
|---|----------|-----|---------|------:|------:|----------------|---------------------|
| 1 | Malagasy | mlg | Madagascar | 61.32 | 20287 | 18900/746/641 | `malagasy_mlg` |
| 2 | Kabuverdianu | kea | Cabo Verde | 58.07 | 19728 | 18156/699/873 | `kabuverdianu_kea` |
| 3 | Shona | sna | Zimbabwe | 53.30 | 17996 | 15954/1161/881 | `shona_sna` |
| 4 | Kabiye | kbp | Togo | 53.19 | 18211 | 16328/825/1058 | `kabiye_kbp` |
| 5 | Twi | twi | Ghana | 50.32 | 17140 | 15794/828/518 | `twi_twi` |
| 6 | Bassa (Cameroon) | bas | Cameroon | 48.58 | 16269 | 14981/620/668 | `bassa_cameroon_bas` |
| 7 | Mauritian Creole | mfe | Mauritius | 43.57 | 15088 | 13115/822/1151 | `mauritian_creole_mfe` |
| 8 | Nyaneka | nyk | Angola | 42.94 | 14731 | 13430/858/443 | `nyaneka_nyk` |
| 9 | Gun | guw | Benin | 39.98 | 13340 | 12157/494/689 | `gun_guw` |
| 10 | Swahili | swa | Tanzania | 38.73 | 13101 | 11781/703/617 | `swahili_swa` |
| 11 | Kiluba | lub | DR Congo | 38.31 | 12949 | 11553/679/717 | `kiluba_lub` |
| 12 | Igbo | ibo | Nigeria | 38.07 | 12639 | 11379/556/704 | `igbo_ibo` |
| 13 | Zulu | zul | South Africa | 36.98 | 12434 | 11540/571/323 | `zulu_zul` |
| 14 | Changana (Mozambique) | tso | Mozambique | 35.77 | 12394 | 11369/478/547 | `changana_mozambique_tso` |
| 15 | Kirundi | run | Burundi | 35.62 | 12399 | 11116/550/733 | `kirundi_run` |
| 16 | Kinyarwanda | kin | Rwanda | 35.57 | 12012 | 10517/606/889 | `kinyarwanda_kin` |
| 17 | Tsonga | tso | South Africa | 35.25 | 11838 | 10710/682/446 | `tsonga_tso` |
| 18 | Ga | gaa | Ghana | 35.12 | 11939 | 10703/632/604 | `ga_gaa` |
| 19 | Amharic | amh | Ethiopia | 33.59 | 11415 | 10340/603/472 | `amharic_amh` |
| 20 | Fon | fon | Benin | 33.19 | 10912 | 10039/459/414 | `fon_fon` |
| 21 | Xhosa | xho | South Africa | 33.15 | 11485 | 10073/658/754 | `xhosa_xho` |
| 22 | Otetela | tll | DR Congo | 31.53 | 11301 | 10316/463/522 | `otetela_tll` |
| 23 | Lingala | lin | DR Congo | 30.71 | 10080 | 9076/414/590 | `lingala_lin` |
| 24 | Ewe | ewe | Ghana | 29.41 | 9760 | 8633/611/516 | `ewe_ewe` |
| 25 | Chitumbuka | tum | Malawi | 29.31 | 10365 | 9303/623/439 | `chitumbuka_tum` |
| 26 | Ronga | rng | Mozambique | 29.11 | 10285 | 9138/642/505 | `ronga_rng` |
| 27 | Kamba | kam | Kenya | 28.86 | 9567 | 8603/348/616 | `kamba_kam` |
| 28 | Kikuyu | kik | Kenya | 28.82 | 9640 | 8808/413/419 | `kikuyu_kik` |
| 29 | Hausa | hau | Nigeria | 28.37 | 9367 | 8364/509/494 | `hausa_hau` |
| 30 | Macua | vmw | Mozambique | 28.02 | 10144 | 8966/682/496 | `macua_vmw` |
| 31 | Kongo | kon | DR Congo | 27.82 | 9980 | 8828/474/678 | `kongo_kon` |
| 32 | Isoko | iso | Nigeria | 26.84 | 9474 | 8544/447/483 | `isoko_iso` |
| 33 | Krio | kri | Sierra Leone | 26.16 | 9230 | 8293/553/384 | `krio_kri` |
| 34 | Edo | bin | Nigeria | 26.04 | 9028 | 8323/346/359 | `edo_bin` |
| 35 | Urhobo | urh | Nigeria | 25.67 | 8944 | 7946/447/551 | `urhobo_urh` |
| 36 | Oromo | orm | Ethiopia | 25.66 | 8933 | 7778/557/598 | `oromo_orm` |
| 37 | Esan | ish | Nigeria | 24.19 | 8499 | 7356/708/435 | `esan_ish` |
| 38 | Frafra | gur | Ghana | 23.82 | 7960 | 7140/398/422 | `frafra_gur` |
| 39 | Sesotho (South Africa) | sot | South Africa | 23.32 | 8149 | 7583/332/234 | `sesotho_south_africa_sot` |
| 40 | Sena | seh | Mozambique | 22.84 | 7849 | 7019/341/489 | `sena_seh` |
| 41 | Sesotho (Lesotho) | sot | South Africa | 22.76 | 7944 | 7052/390/502 | `sesotho_lesotho_sot` |
| 42 | Lenje | leh | Zambia | 22.52 | 8099 | 7198/412/489 | `lenje_leh` |
| 43 | Liberian English | lir | Liberia | 22.06 | 7733 | 7112/335/286 | `liberian_english_lir` |
| 44 | Fante | fat | Ghana | 22.04 | 7631 | 6942/277/412 | `fante_fat` |
| 45 | Dagaare | dga | Ghana | 21.53 | 7477 | 6575/375/527 | `dagaare_dga` |
| 46 | Nzema | nzi | Ghana | 21.14 | 7319 | 6413/532/374 | `nzema_nzi` |
| 47 | Pidgin (West Africa) | wes | Cameroon | 21.09 | 7405 | 6785/227/393 | `pidgin_west_africa_wes` |
| 48 | Kisi | kss | Liberia | 20.99 | 7739 | 6967/377/395 | `kisi_kss` |
| 49 | Moore | mos | Burkina Faso | 20.51 | 6785 | 6293/196/296 | `moore_mos` |
| 50 | Ahanta | aha | Ghana | 20.30 | 6908 | 6038/441/429 | `ahanta_aha` |
| 51 | Douala | dua | Cameroon | 20.12 | 6521 | 5832/145/544 | `douala_dua` |
| 52 | Luo | luo | Kenya | 19.66 | 6829 | 5959/367/503 | `luo_luo` |
| 53 | Sepedi | nso | South Africa | 19.59 | 7016 | 6435/396/185 | `sepedi_nso` |
| 54 | Bissau Guinean Creole | pov | Guinea-Bissau | 19.35 | 6573 | 5899/375/299 | `bissau_guinean_creole_pov` |
| 55 | Swahili (Congo) | swc | DR Congo | 18.67 | 6493 | 5670/503/320 | `swahili_congo_swc` |
| 56 | Sehwi | sfw | Ghana | 17.48 | 6024 | 5488/199/337 | `sehwi_sfw` |
| 57 | Runyankore | nyn | Uganda | 17.10 | 5605 | 4736/432/437 | `runyankore_nyn` |
| 58 | Boulou | bum | Cameroon | 16.93 | 5692 | 5153/231/308 | `boulou_bum` |
| 59 | Kwanyama | kua | Namibia | 16.66 | 6107 | 5467/239/401 | `kwanyama_kua` |
| 60 | Jula | dyu | Burkina Faso | 15.85 | 5486 | 5019/258/209 | `jula_dyu` |
| 61 | Tshwa | tsc | Mozambique | 14.99 | 5221 | 4651/209/361 | `tshwa_tsc` |
| 62 | Yoruba | yor | Nigeria | 14.19 | 4960 | 4504/216/240 | `yoruba_yor` |
| 63 | Setswana | tsn | South Africa | 13.84 | 4828 | 4232/344/252 | `setswana_tsn` |
| 64 | Kikongo ya Leta | ktu | DR Congo | 13.81 | 4754 | 4232/262/260 | `kikongo_ya_leta_ktu` |
| 65 | Chichewa | nya | Malawi | 12.81 | 3884 | 3407/231/246 | `chichewa_nya` |
| 66 | Sango | sag | Central African Republic | 12.60 | 4193 | 3645/261/287 | `sango_sag` |
| 67 | Mashi | shr | DR Congo | 12.41 | 4263 | 3855/273/135 | `mashi_shr` |
| 68 | Seychelles Creole | crs | Seychelles | 12.16 | 4269 | 3983/72/214 | `seychelles_creole_crs` |
| 69 | Fang | fan | Equatorial Guinea | 11.68 | 3969 | 3630/184/155 | `fang_fan` |
| 70 | Luganda | lug | Uganda | 11.25 | 3617 | 3237/75/305 | `luganda_lug` |
| 71 | Kituba | ktu | DR Congo | 11.24 | 3935 | 3639/195/101 | `kituba_ktu` |
| 72 | Tshiluba | lua | DR Congo | 11.24 | 3532 | 3234/158/140 | `tshiluba_lua` |
| 73 | Chopi | cce | Mozambique | 10.80 | 3789 | 3414/144/231 | `chopi_cce` |
| 74 | Cibemba | bem | Zambia | 10.56 | 3823 | 3333/170/320 | `cibemba_bem` |
| 75 | Ngangela | nba | Angola | 10.28 | 3638 | 3348/166/124 | `ngangela_nba` |
| 76 | Ndau (Western) | ndc | Mozambique | 10.22 | 3621 | 3235/239/147 | `ndau_western_ndc` |
| 77 | Phimbi | phm | Mozambique | 10.00 | 3561 | 3172/121/268 | `phimbi_phm` |
| 78 | Swati | ssw | South Africa | 9.42 | 3236 | 2767/170/299 | `swati_ssw` |
| 79 | Toupouri | tui | Cameroon | 8.96 | 3070 | 2588/292/190 | `toupouri_tui` |
| 80 | Venda | ven | South Africa | 8.65 | 2990 | 2721/135/134 | `venda_ven` |
| 81 | Nyungwe | nyu | Mozambique | 8.35 | 2976 | 2744/63/169 | `nyungwe_nyu` |
| 82 | Kimbundu | kmb | Angola | 8.32 | 2894 | 2468/249/177 | `kimbundu_kmb` |
| 83 | Kinande | nnb | DR Congo | 8.15 | 2871 | 2637/44/190 | `kinande_nnb` |
| 84 | Nsenga (Mozambique) | nse | Mozambique | 8.09 | 2865 | 2385/313/167 | `nsenga_mozambique_nse` |
| 85 | Baoule | bci | Côte d'Ivoire | 8.08 | 2657 | 2307/125/225 | `baoule_bci` |
| 86 | Gokana | gkn | Nigeria | 7.98 | 2808 | 2567/179/62 | `gokana_gkn` |
| 87 | Ndebele (Zimbabwe) | nde | Zimbabwe | 7.62 | 2552 | 2308/81/163 | `ndebele_zimbabwe_nde` |
| 88 | Havu | hav | DR Congo | 7.59 | 2619 | 2447/172/0 | `havu_hav` |
| 89 | Ibinda | yom | DR Congo | 7.46 | 2672 | 2508/59/105 | `ibinda_yom` |
| 90 | Dinka | din | South Sudan | 6.84 | 2496 | 2178/99/219 | `dinka_din` |
| 91 | Itsekiri | its | Nigeria | 6.78 | 2387 | 2387/0/0 | `itsekiri_its` |
| 92 | Kwangali | kwn | Namibia | 6.75 | 2258 | 2139/81/38 | `kwangali_kwn` |
| 93 | Chitonga | toi | Zambia | 6.61 | 2487 | 2220/195/72 | `chitonga_toi` |
| 94 | Bassa (Liberia) | bsq | Liberia | 6.07 | 2284 | 2135/12/137 | `bassa_liberia_bsq` |
| 95 | Cinyanja | nya | Malawi | 5.79 | 1883 | 1765/42/76 | `cinyanja_nya` |
| 96 | Ndonga | ndo | Namibia | 5.77 | 1920 | 1745/63/112 | `ndonga_ndo` |
| 97 | Aja | ajg | Benin | 5.71 | 1942 | 1765/105/72 | `aja_ajg` |
| 98 | Kpelle | xpe | Liberia | 5.70 | 1933 | 1743/112/78 | `kpelle_xpe` |
| 99 | Réunion Creole | rcf | Réunion | 5.25 | 1803 | 1718/23/62 | `r_union_creole_rcf` |
| 100 | Ndebele | nbl | South Africa | 5.15 | 1985 | 1819/134/32 | `ndebele_nbl` |
| 101 | Abbey | aba | Côte d'Ivoire | 5.03 | 1776 | 1574/166/36 | `abbey_aba` |
| 102 | Yombe | yom | DR Congo | 4.88 | 1634 | 1390/149/95 | `yombe_yom` |
| 103 | Kikongo | kwy | Angola | 4.87 | 1769 | 1623/103/43 | `kikongo_kwy` |
| 104 | Umbundu | umb | Angola | 4.70 | 1630 | 1340/227/63 | `umbundu_umb` |
| 105 | Chiyao | yao | Mozambique | 4.61 | 1647 | 1423/103/121 | `chiyao_yao` |
| 106 | Loma | lom | Liberia | 4.52 | 1540 | 1350/41/149 | `loma_lom` |
| 107 | Wolaita | wal | Ethiopia | 4.28 | 1476 | 1250/133/93 | `wolaita_wal` |
| 108 | Chitonga (Malawi) | tog | Malawi | 4.16 | 1480 | 1363/65/52 | `chitonga_malawi_tog` |
| 109 | Tiv | tiv | Nigeria | 4.03 | 1374 | 1220/119/35 | `tiv_tiv` |
| 110 | Lari | ldi | Congo | 3.85 | 1359 | 1284/42/33 | `lari_ldi` |
| 111 | Meru | mer | Kenya | 3.83 | 1277 | 1139/138/0 | `meru_mer` |
| 112 | Ewondo | ewo | Cameroon | 3.71 | 1305 | 1134/32/139 | `ewondo_ewo` |
| 113 | Kabyle | kab | Algeria | 3.65 | 1188 | 1068/64/56 | `kabyle_kab` |
| 114 | Khana | ogo | Nigeria | 3.62 | 1256 | 921/139/196 | `khana_ogo` |
| 115 | Gitonga | toh | Mozambique | 3.60 | 1319 | 1228/4/87 | `gitonga_toh` |
| 116 | Tewe | twx | Mozambique | 3.39 | 1251 | 1076/148/27 | `tewe_twx` |
| 117 | Dangme | ada | Ghana | 3.37 | 1177 | 941/154/82 | `dangme_ada` |
| 118 | Ndau | ndc | Mozambique | 3.34 | 1231 | 1147/68/16 | `ndau_ndc` |
| 119 | Guéré | gxx | Côte d'Ivoire | 3.01 | 1057 | 953/67/37 | `gu_r_gxx` |
| 120 | Wolof | wol | Senegal | 2.50 | 846 | 803/10/33 | `wolof_wol` |
| 121 | Damara | naq | Namibia | 2.46 | 789 | 698/62/29 | `damara_naq` |
| 122 | Swahili (Katanga) | swc | DR Congo | 2.40 | 866 | 798/32/36 | `swahili_katanga_swc` |
| 123 | Yacouba | daf | Côte d'Ivoire | 2.26 | 787 | 656/0/131 | `yacouba_daf` |
| 124 | Manyawa | mny | Mozambique | 1.94 | 697 | 697/0/0 | `manyawa_mny` |
| 125 | Makhuwa-Marrevone | xmc | Mozambique | 1.87 | 704 | 635/41/28 | `makhuwa_marrevone_xmc` |
| 126 | Makhuwa-Meetto | mgh | Mozambique | 1.83 | 671 | 552/85/34 | `makhuwa_meetto_mgh` |
| 127 | Cinamwanga | mwn | Zambia | 1.66 | 553 | 500/53/0 | `cinamwanga_mwn` |
| 128 | Chitonga (Zimbabwe) | toi | Zimbabwe | 1.41 | 486 | 452/0/34 | `chitonga_zimbabwe_toi` |
| 129 | Attié | ati | Côte d'Ivoire | 1.40 | 482 | 477/5/0 | `atti_ati` |
| 130 | Lunda | lun | Zambia | 1.11 | 423 | 322/101/0 | `lunda_lun` |
| 131 | Lomwe | ngl | Mozambique | 1.07 | 378 | 373/5/0 | `lomwe_ngl` |
| 132 | Chuabo | chw | Mozambique | 1.03 | 390 | 373/0/17 | `chuabo_chw` |
| 133 | Mambwe-Lungu | mgr | Zambia | 0.93 | 331 | 264/34/33 | `mambwe_lungu_mgr` |
| 134 | Ijaw | ijc | Nigeria | 0.88 | 312 | 292/0/20 | `ijaw_ijc` |
| 135 | Ngbandi (Northern) | ngb | DR Congo | 0.78 | 255 | 222/33/0 | `ngbandi_northern_ngb` |
| 136 | Makhuwa-Shirima | vmk | Mozambique | 0.77 | 277 | 277/0/0 | `makhuwa_shirima_vmk` |
| 137 | Herero | her | Namibia | 0.61 | 192 | 192/0/0 | `herero_her` |
| 138 | Chokwe | cjk | Angola | 0.56 | 177 | 169/0/8 | `chokwe_cjk` |
| 139 | Taabwa | tap | DR Congo | 0.56 | 195 | 163/0/32 | `taabwa_tap` |
| 140 | Kisonge | sop | DR Congo | 0.40 | 141 | 141/0/0 | `kisonge_sop` |
| 141 | Kanyok | kny | DR Congo | 0.28 | 109 | 109/0/0 | `kanyok_kny` |
| 142 | Luvale | lue | Zambia | 0.07 | 20 | 20/0/0 | `luvale_lue` |

## License

CC-BY-4.0
