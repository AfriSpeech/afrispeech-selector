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

<details>
<summary><b>Expand the full 369-config catalog</b> &mdash; 308 languages, ~7,854 hours, 1,938,892 clips</summary>

<table>
<thead><tr><th>#</th><th>Language</th><th>ISO</th><th>Country</th><th>Hours</th><th>Clips</th><th>Train/Val/Test</th><th><code>--languages</code> value</th><th>Source</th></tr></thead>
<tbody>
<tr><td>1</td><td>Tigrinya</td><td>tir</td><td>Eritrea</td><td>143.52</td><td>50,315</td><td>40,856/4,425/5,034</td><td><code>tir_asr</code></td><td>waxal</td></tr>
<tr><td>2</td><td>Chichewa</td><td>nya</td><td>Malawi</td><td>139.02</td><td>30,322</td><td>28,805/0/1,517</td><td><code>Chichewa</code></td><td>open-bible</td></tr>
<tr><td>3</td><td>Wolaytta</td><td>wal</td><td>Ethiopia</td><td>134.89</td><td>47,291</td><td>41,833/2,444/3,014</td><td><code>wal_asr</code></td><td>waxal</td></tr>
<tr><td>4</td><td>Sidamo</td><td>sid</td><td>Ethiopia</td><td>129.51</td><td>45,404</td><td>38,752/3,091/3,561</td><td><code>sid_asr</code></td><td>waxal</td></tr>
<tr><td>5</td><td>Oromo</td><td>orm</td><td>Ethiopia</td><td>128.49</td><td>45,045</td><td>38,185/3,078/3,782</td><td><code>orm_asr</code></td><td>waxal</td></tr>
<tr><td>6</td><td>Amharic</td><td>amh</td><td>Ethiopia</td><td>126.51</td><td>44,354</td><td>38,022/2,912/3,420</td><td><code>amh_asr</code></td><td>waxal</td></tr>
<tr><td>7</td><td>Igbo</td><td>ibo</td><td>Nigeria</td><td>119.02</td><td>30,011</td><td>28,510/0/1,501</td><td><code>Igbo</code></td><td>open-bible</td></tr>
<tr><td>8</td><td>Gamo</td><td>gmv</td><td>Ethiopia</td><td>118.78</td><td>30,200</td><td>28,690/0/1,510</td><td><code>Gamo</code></td><td>open-bible</td></tr>
<tr><td>9</td><td>North Ndebele</td><td>nde</td><td>Botswana</td><td>111.17</td><td>30,161</td><td>28,652/0/1,509</td><td><code>Ndebele</code></td><td>open-bible</td></tr>
<tr><td>10</td><td>Lingala</td><td>lin</td><td>DR Congo</td><td>108.68</td><td>28,790</td><td>27,350/0/1,440</td><td><code>Lingala</code></td><td>open-bible</td></tr>
<tr><td>11</td><td>Kikuyu</td><td>kik</td><td>Kenya</td><td>107.76</td><td>30,722</td><td>29,185/0/1,537</td><td><code>Kikuyu</code></td><td>open-bible</td></tr>
<tr><td>12</td><td>Dawro</td><td>dwr</td><td>Ethiopia</td><td>99.94</td><td>29,579</td><td>28,100/0/1,479</td><td><code>Dawro</code></td><td>open-bible</td></tr>
<tr><td>13</td><td>Yoruba</td><td>yor</td><td>Benin</td><td>95.84</td><td>30,625</td><td>29,093/0/1,532</td><td><code>Yoruba</code></td><td>open-bible</td></tr>
<tr><td>14</td><td>Oromo</td><td>orm</td><td>Ethiopia</td><td>93.68</td><td>30,413</td><td>28,892/0/1,521</td><td><code>Oromo</code></td><td>open-bible</td></tr>
<tr><td>15</td><td>Swahili (macrolanguage)</td><td>swa</td><td>Burundi</td><td>93.06</td><td>30,634</td><td>29,102/0/1,532</td><td><code>Swahili</code></td><td>open-bible</td></tr>
<tr><td>16</td><td>Ganda</td><td>lug</td><td>Uganda</td><td>91.11</td><td>30,440</td><td>28,918/0/1,522</td><td><code>Luganda</code></td><td>open-bible</td></tr>
<tr><td>17</td><td>Ewe</td><td>ewe</td><td>Ghana</td><td>86.24</td><td>30,164</td><td>28,655/0/1,509</td><td><code>Ewe</code></td><td>open-bible</td></tr>
<tr><td>18</td><td>Hausa</td><td>hau</td><td>Burkina Faso</td><td>83.64</td><td>30,717</td><td>29,181/0/1,536</td><td><code>Hausa</code></td><td>open-bible</td></tr>
<tr><td>19</td><td>Shona</td><td>sna</td><td>Botswana</td><td>82.42</td><td>30,685</td><td>29,150/0/1,535</td><td><code>Shona</code></td><td>open-bible</td></tr>
<tr><td>20</td><td>Twi</td><td>twi</td><td>Ghana</td><td>78.92</td><td>30,565</td><td>29,036/0/1,529</td><td><code>Twi (Asante)</code></td><td>open-bible</td></tr>
<tr><td>21</td><td>Gofa</td><td>gof</td><td>Ethiopia</td><td>78.76</td><td>30,388</td><td>28,868/0/1,520</td><td><code>Gofa</code></td><td>open-bible</td></tr>
<tr><td>22</td><td>Fulah</td><td>ful</td><td>Nigeria</td><td>68.12</td><td>23,881</td><td>19,132/2,358/2,391</td><td><code>ful_asr</code></td><td>waxal</td></tr>
<tr><td>23</td><td>Luo (Kenya and Tanzania)</td><td>luo</td><td>Kenya</td><td>67.36</td><td>30,429</td><td>28,907/0/1,522</td><td><code>Dholuo</code></td><td>open-bible</td></tr>
<tr><td>24</td><td>Malagasy</td><td>mlg</td><td>Madagascar</td><td>65.85</td><td>23,086</td><td>18,526/2,270/2,290</td><td><code>mlg_asr</code></td><td>waxal</td></tr>
<tr><td>25</td><td>Malagasy</td><td>mlg</td><td>Madagascar</td><td>61.32</td><td>20,287</td><td>18,900/746/641</td><td><code>malagasy_mlg</code></td><td>afrispeech</td></tr>
<tr><td>26</td><td>Kabuverdianu</td><td>kea</td><td>Cabo Verde</td><td>58.07</td><td>19,728</td><td>18,156/699/873</td><td><code>kabuverdianu_kea</code></td><td>afrispeech</td></tr>
<tr><td>27</td><td>Twi</td><td>twi</td><td>Ghana</td><td>55.28</td><td>30,270</td><td>28,756/0/1,514</td><td><code>Twi (Akuapem)</code></td><td>open-bible</td></tr>
<tr><td>28</td><td>Southern Dagaare</td><td>dga</td><td>Ghana</td><td>53.84</td><td>18,874</td><td>15,071/1,893/1,910</td><td><code>dga_asr</code></td><td>waxal</td></tr>
<tr><td>29</td><td>Ewe</td><td>ewe</td><td>Ghana</td><td>53.80</td><td>18,861</td><td>15,054/1,916/1,891</td><td><code>ewe_asr</code></td><td>waxal</td></tr>
<tr><td>30</td><td>Shona</td><td>sna</td><td>Zimbabwe</td><td>53.30</td><td>17,996</td><td>15,954/1,161/881</td><td><code>shona_sna</code></td><td>afrispeech</td></tr>
<tr><td>31</td><td>Kabiye</td><td>kbp</td><td>Togo</td><td>53.19</td><td>18,211</td><td>16,328/825/1,058</td><td><code>kabiye_kbp</code></td><td>afrispeech</td></tr>
<tr><td>32</td><td>Eastern Yiddish (Hebr script)</td><td>ydd</td><td>South Africa</td><td>51.80</td><td>3,000</td><td>1,950/300/750</td><td><code>ydd_Hebr</code></td><td>omnilingual</td></tr>
<tr><td>33</td><td>Northern Betsimisaraka Malagasy</td><td>bmm</td><td>Madagascar</td><td>51.76</td><td>2,998</td><td>2,998/0/0</td><td><code>bmm_Latn</code></td><td>omnilingual</td></tr>
<tr><td>34</td><td>Orma</td><td>orc</td><td>Kenya</td><td>51.76</td><td>2,998</td><td>1,500/900/598</td><td><code>orc_Latn</code></td><td>omnilingual</td></tr>
<tr><td>35</td><td>Southern Betsimisaraka Malagasy</td><td>bzc</td><td>Madagascar</td><td>51.75</td><td>2,997</td><td>2,997/0/0</td><td><code>bzc_Latn</code></td><td>omnilingual</td></tr>
<tr><td>36</td><td>Tesaka Malagasy</td><td>tkg</td><td>Madagascar</td><td>51.75</td><td>2,997</td><td>2,099/598/300</td><td><code>tkg_Latn</code></td><td>omnilingual</td></tr>
<tr><td>37</td><td>Plateau Malagasy</td><td>plt</td><td>Madagascar</td><td>51.71</td><td>2,995</td><td>2,995/0/0</td><td><code>plt_Latn</code></td><td>omnilingual</td></tr>
<tr><td>38</td><td>Lingala</td><td>lin</td><td>DR Congo</td><td>51.66</td><td>18,110</td><td>14,400/1,844/1,866</td><td><code>lin_asr</code></td><td>waxal</td></tr>
<tr><td>39</td><td>Pular</td><td>fuf</td><td>Guinea</td><td>51.51</td><td>2,983</td><td>2,983/0/0</td><td><code>fuf_Latn</code></td><td>omnilingual</td></tr>
<tr><td>40</td><td>Egyptian Arabic (Arab script)</td><td>arz</td><td>Egypt</td><td>51.45</td><td>2,980</td><td>2,384/298/298</td><td><code>arz_Arab</code></td><td>omnilingual</td></tr>
<tr><td>41</td><td>Libyan Arabic (Arab script)</td><td>ayl</td><td>Egypt</td><td>51.45</td><td>2,980</td><td>2,384/298/298</td><td><code>ayl_Arab</code></td><td>omnilingual</td></tr>
<tr><td>42</td><td>Ikposo</td><td>kpo</td><td>Ghana</td><td>51.40</td><td>18,020</td><td>14,415/1,760/1,845</td><td><code>kpo_asr</code></td><td>waxal</td></tr>
<tr><td>43</td><td>Bagirmi Fulfulde</td><td>fui</td><td>Central African Republic</td><td>51.18</td><td>2,964</td><td>2,964/0/0</td><td><code>fui_Latn</code></td><td>omnilingual</td></tr>
<tr><td>44</td><td>Dagbani</td><td>dag</td><td>Ghana</td><td>50.83</td><td>17,819</td><td>14,231/1,750/1,838</td><td><code>dag_asr</code></td><td>waxal</td></tr>
<tr><td>45</td><td>Tunisian Arabic (Arab script)</td><td>aeb</td><td>Algeria</td><td>50.50</td><td>2,925</td><td>2,925/0/0</td><td><code>aeb_Arab</code></td><td>omnilingual</td></tr>
<tr><td>46</td><td>Twi</td><td>twi</td><td>Ghana</td><td>50.32</td><td>17,140</td><td>15,794/828/518</td><td><code>twi_twi</code></td><td>afrispeech</td></tr>
<tr><td>47</td><td>Shona</td><td>sna</td><td>Zimbabwe</td><td>50.16</td><td>17,585</td><td>14,109/1,727/1,749</td><td><code>sna_asr</code></td><td>waxal</td></tr>
<tr><td>48</td><td>Bassa (Cameroon)</td><td>bas</td><td>Cameroon</td><td>48.58</td><td>16,269</td><td>14,981/620/668</td><td><code>bassa_cameroon_bas</code></td><td>afrispeech</td></tr>
<tr><td>49</td><td>Masikoro Malagasy</td><td>msh</td><td>Madagascar</td><td>46.57</td><td>2,697</td><td>2,697/0/0</td><td><code>msh_Latn</code></td><td>omnilingual</td></tr>
<tr><td>50</td><td>Antankarana Malagasy</td><td>xmv</td><td>Madagascar</td><td>43.96</td><td>2,546</td><td>1,803/450/293</td><td><code>xmv_Latn</code></td><td>omnilingual</td></tr>
<tr><td>51</td><td>Mauritian Creole</td><td>mfe</td><td>Mauritius</td><td>43.57</td><td>15,088</td><td>13,115/822/1,151</td><td><code>mauritian_creole_mfe</code></td><td>afrispeech</td></tr>
<tr><td>52</td><td>Nyaneka</td><td>nyk</td><td>Angola</td><td>42.94</td><td>14,731</td><td>13,430/858/443</td><td><code>nyaneka_nyk</code></td><td>afrispeech</td></tr>
<tr><td>53</td><td>Pulaar</td><td>fuc</td><td>Gambia</td><td>42.65</td><td>2,470</td><td>2,470/0/0</td><td><code>fuc_Latn</code></td><td>omnilingual</td></tr>
<tr><td>54</td><td>Gun</td><td>guw</td><td>Benin</td><td>39.98</td><td>13,340</td><td>12,157/494/689</td><td><code>gun_guw</code></td><td>afrispeech</td></tr>
<tr><td>55</td><td>Swahili</td><td>swa</td><td>Tanzania</td><td>38.73</td><td>13,101</td><td>11,781/703/617</td><td><code>swahili_swa</code></td><td>afrispeech</td></tr>
<tr><td>56</td><td>Kiluba</td><td>lub</td><td>DR Congo</td><td>38.31</td><td>12,949</td><td>11,553/679/717</td><td><code>kiluba_lub</code></td><td>afrispeech</td></tr>
<tr><td>57</td><td>Igbo</td><td>ibo</td><td>Nigeria</td><td>38.07</td><td>12,639</td><td>11,379/556/704</td><td><code>igbo_ibo</code></td><td>afrispeech</td></tr>
<tr><td>58</td><td>Zulu</td><td>zul</td><td>South Africa</td><td>36.98</td><td>12,434</td><td>11,540/571/323</td><td><code>zulu_zul</code></td><td>afrispeech</td></tr>
<tr><td>59</td><td>Akan</td><td>aka</td><td>Ghana</td><td>36.37</td><td>12,752</td><td>10,107/1,123/1,522</td><td><code>aka_asr</code></td><td>waxal</td></tr>
<tr><td>60</td><td>Changana (Mozambique)</td><td>tso</td><td>Mozambique</td><td>35.77</td><td>12,394</td><td>11,369/478/547</td><td><code>changana_mozambique_tso</code></td><td>afrispeech</td></tr>
<tr><td>61</td><td>Kirundi</td><td>run</td><td>Burundi</td><td>35.62</td><td>12,399</td><td>11,116/550/733</td><td><code>kirundi_run</code></td><td>afrispeech</td></tr>
<tr><td>62</td><td>Kinyarwanda</td><td>kin</td><td>Rwanda</td><td>35.57</td><td>12,012</td><td>10,517/606/889</td><td><code>kinyarwanda_kin</code></td><td>afrispeech</td></tr>
<tr><td>63</td><td>Tsonga</td><td>tso</td><td>South Africa</td><td>35.25</td><td>11,838</td><td>10,710/682/446</td><td><code>tsonga_tso</code></td><td>afrispeech</td></tr>
<tr><td>64</td><td>Ga</td><td>gaa</td><td>Ghana</td><td>35.12</td><td>11,939</td><td>10,703/632/604</td><td><code>ga_gaa</code></td><td>afrispeech</td></tr>
<tr><td>65</td><td>Bara Malagasy</td><td>bhr</td><td>Madagascar</td><td>33.70</td><td>1,952</td><td>1,952/0/0</td><td><code>bhr_Latn</code></td><td>omnilingual</td></tr>
<tr><td>66</td><td>Amharic</td><td>amh</td><td>Ethiopia</td><td>33.59</td><td>11,415</td><td>10,340/603/472</td><td><code>amharic_amh</code></td><td>afrispeech</td></tr>
<tr><td>67</td><td>Fon</td><td>fon</td><td>Benin</td><td>33.19</td><td>10,912</td><td>10,039/459/414</td><td><code>fon_fon</code></td><td>afrispeech</td></tr>
<tr><td>68</td><td>Moroccan Arabic (Arab script)</td><td>ary</td><td>Algeria</td><td>33.19</td><td>1,922</td><td>1,488/0/434</td><td><code>ary_Arab</code></td><td>omnilingual</td></tr>
<tr><td>69</td><td>Xhosa</td><td>xho</td><td>South Africa</td><td>33.15</td><td>11,485</td><td>10,073/658/754</td><td><code>xhosa_xho</code></td><td>afrispeech</td></tr>
<tr><td>70</td><td>Otetela</td><td>tll</td><td>DR Congo</td><td>31.53</td><td>11,301</td><td>10,316/463/522</td><td><code>otetela_tll</code></td><td>afrispeech</td></tr>
<tr><td>71</td><td>Tsimihety Malagasy</td><td>xmw</td><td>Madagascar</td><td>31.08</td><td>1,800</td><td>1,800/0/0</td><td><code>xmw_Latn</code></td><td>omnilingual</td></tr>
<tr><td>72</td><td>Tanosy Malagasy</td><td>txy</td><td>Madagascar</td><td>30.99</td><td>1,795</td><td>898/448/449</td><td><code>txy_Latn</code></td><td>omnilingual</td></tr>
<tr><td>73</td><td>Lingala</td><td>lin</td><td>DR Congo</td><td>30.71</td><td>10,080</td><td>9,076/414/590</td><td><code>lingala_lin</code></td><td>afrispeech</td></tr>
<tr><td>74</td><td>Ewe</td><td>ewe</td><td>Ghana</td><td>29.41</td><td>9,760</td><td>8,633/611/516</td><td><code>ewe_ewe</code></td><td>afrispeech</td></tr>
<tr><td>75</td><td>Chitumbuka</td><td>tum</td><td>Malawi</td><td>29.31</td><td>10,365</td><td>9,303/623/439</td><td><code>chitumbuka_tum</code></td><td>afrispeech</td></tr>
<tr><td>76</td><td>Ronga</td><td>rng</td><td>Mozambique</td><td>29.11</td><td>10,285</td><td>9,138/642/505</td><td><code>ronga_rng</code></td><td>afrispeech</td></tr>
<tr><td>77</td><td>Kamba</td><td>kam</td><td>Kenya</td><td>28.86</td><td>9,567</td><td>8,603/348/616</td><td><code>kamba_kam</code></td><td>afrispeech</td></tr>
<tr><td>78</td><td>Kikuyu</td><td>kik</td><td>Kenya</td><td>28.82</td><td>9,640</td><td>8,808/413/419</td><td><code>kikuyu_kik</code></td><td>afrispeech</td></tr>
<tr><td>79</td><td>Hausa</td><td>hau</td><td>Nigeria</td><td>28.37</td><td>9,367</td><td>8,364/509/494</td><td><code>hausa_hau</code></td><td>afrispeech</td></tr>
<tr><td>80</td><td>Macua</td><td>vmw</td><td>Mozambique</td><td>28.02</td><td>10,144</td><td>8,966/682/496</td><td><code>macua_vmw</code></td><td>afrispeech</td></tr>
<tr><td>81</td><td>Kongo</td><td>kon</td><td>DR Congo</td><td>27.82</td><td>9,980</td><td>8,828/474/678</td><td><code>kongo_kon</code></td><td>afrispeech</td></tr>
<tr><td>82</td><td>Matengo</td><td>mgv</td><td>Tanzania</td><td>27.32</td><td>7,792</td><td>7,402/0/390</td><td><code>Matengo</code></td><td>open-bible</td></tr>
<tr><td>83</td><td>Isoko</td><td>iso</td><td>Nigeria</td><td>26.84</td><td>9,474</td><td>8,544/447/483</td><td><code>isoko_iso</code></td><td>afrispeech</td></tr>
<tr><td>84</td><td>Krio</td><td>kri</td><td>Sierra Leone</td><td>26.16</td><td>9,230</td><td>8,293/553/384</td><td><code>krio_kri</code></td><td>afrispeech</td></tr>
<tr><td>85</td><td>Edo</td><td>bin</td><td>Nigeria</td><td>26.04</td><td>9,028</td><td>8,323/346/359</td><td><code>edo_bin</code></td><td>afrispeech</td></tr>
<tr><td>86</td><td>Bomu</td><td>bmq</td><td>Burkina Faso</td><td>25.90</td><td>1,500</td><td>1,500/0/0</td><td><code>bmq_Latn</code></td><td>omnilingual</td></tr>
<tr><td>87</td><td>Maasina Fulfulde</td><td>ffm</td><td>Burkina Faso</td><td>25.90</td><td>1,500</td><td>1,500/0/0</td><td><code>ffm_Latn</code></td><td>omnilingual</td></tr>
<tr><td>88</td><td>Sakalava Malagasy</td><td>skg</td><td>Madagascar</td><td>25.88</td><td>1,499</td><td>1,499/0/0</td><td><code>skg_Latn</code></td><td>omnilingual</td></tr>
<tr><td>89</td><td>Algerian Arabic (Arab script)</td><td>arq</td><td>Algeria</td><td>25.71</td><td>1,489</td><td>1,489/0/0</td><td><code>arq_Arab</code></td><td>omnilingual</td></tr>
<tr><td>90</td><td>Urhobo</td><td>urh</td><td>Nigeria</td><td>25.67</td><td>8,944</td><td>7,946/447/551</td><td><code>urhobo_urh</code></td><td>afrispeech</td></tr>
<tr><td>91</td><td>Oromo</td><td>orm</td><td>Ethiopia</td><td>25.66</td><td>8,933</td><td>7,778/557/598</td><td><code>oromo_orm</code></td><td>afrispeech</td></tr>
<tr><td>92</td><td>Central-Eastern Niger Fulfulde</td><td>fuq</td><td>Mali</td><td>25.48</td><td>1,476</td><td>600/431/445</td><td><code>fuq_Latn</code></td><td>omnilingual</td></tr>
<tr><td>93</td><td>Saidi Arabic (Arab script)</td><td>aec</td><td>Egypt</td><td>25.35</td><td>1,468</td><td>1,168/150/150</td><td><code>aec_Arab</code></td><td>omnilingual</td></tr>
<tr><td>94</td><td>Masaaba</td><td>myx</td><td>Uganda</td><td>24.46</td><td>8,574</td><td>6,867/849/858</td><td><code>mas_asr</code></td><td>waxal</td></tr>
<tr><td>95</td><td>Esan</td><td>ish</td><td>Nigeria</td><td>24.19</td><td>8,499</td><td>7,356/708/435</td><td><code>esan_ish</code></td><td>afrispeech</td></tr>
<tr><td>96</td><td>Nyankole</td><td>nyn</td><td>Uganda</td><td>24.17</td><td>8,472</td><td>6,783/831/858</td><td><code>nyn_asr</code></td><td>waxal</td></tr>
<tr><td>97</td><td>Fulah</td><td>ful</td><td>Nigeria</td><td>23.91</td><td>3,878</td><td>3,100/380/398</td><td><code>ful_tts</code></td><td>waxal</td></tr>
<tr><td>98</td><td>Frafra</td><td>gur</td><td>Ghana</td><td>23.82</td><td>7,960</td><td>7,140/398/422</td><td><code>frafra_gur</code></td><td>afrispeech</td></tr>
<tr><td>99</td><td>Sesotho (South Africa)</td><td>sot</td><td>South Africa</td><td>23.32</td><td>8,149</td><td>7,583/332/234</td><td><code>sesotho_south_africa_sot</code></td><td>afrispeech</td></tr>
<tr><td>100</td><td>Sena</td><td>seh</td><td>Mozambique</td><td>22.84</td><td>7,849</td><td>7,019/341/489</td><td><code>sena_seh</code></td><td>afrispeech</td></tr>
<tr><td>101</td><td>Sesotho (Lesotho)</td><td>sot</td><td>South Africa</td><td>22.76</td><td>7,944</td><td>7,052/390/502</td><td><code>sesotho_lesotho_sot</code></td><td>afrispeech</td></tr>
<tr><td>102</td><td>Lenje</td><td>leh</td><td>Zambia</td><td>22.52</td><td>8,099</td><td>7,198/412/489</td><td><code>lenje_leh</code></td><td>afrispeech</td></tr>
<tr><td>103</td><td>Soga</td><td>xog</td><td>Uganda</td><td>22.47</td><td>7,877</td><td>6,219/841/817</td><td><code>sog_asr</code></td><td>waxal</td></tr>
<tr><td>104</td><td>Liberian English</td><td>lir</td><td>Liberia</td><td>22.06</td><td>7,733</td><td>7,112/335/286</td><td><code>liberian_english_lir</code></td><td>afrispeech</td></tr>
<tr><td>105</td><td>Fante</td><td>fat</td><td>Ghana</td><td>22.04</td><td>7,631</td><td>6,942/277/412</td><td><code>fante_fat</code></td><td>afrispeech</td></tr>
<tr><td>106</td><td>Dagaare</td><td>dga</td><td>Ghana</td><td>21.53</td><td>7,477</td><td>6,575/375/527</td><td><code>dagaare_dga</code></td><td>afrispeech</td></tr>
<tr><td>107</td><td>Nzema</td><td>nzi</td><td>Ghana</td><td>21.14</td><td>7,319</td><td>6,413/532/374</td><td><code>nzema_nzi</code></td><td>afrispeech</td></tr>
<tr><td>108</td><td>Pidgin (West Africa)</td><td>wes</td><td>Cameroon</td><td>21.09</td><td>7,405</td><td>6,785/227/393</td><td><code>pidgin_west_africa_wes</code></td><td>afrispeech</td></tr>
<tr><td>109</td><td>Kisi</td><td>kss</td><td>Liberia</td><td>20.99</td><td>7,739</td><td>6,967/377/395</td><td><code>kisi_kss</code></td><td>afrispeech</td></tr>
<tr><td>110</td><td>Borgu Fulfulde</td><td>fue</td><td>Benin</td><td>20.67</td><td>1,197</td><td>1,197/0/0</td><td><code>fue_Latn</code></td><td>omnilingual</td></tr>
<tr><td>111</td><td>Fanti</td><td>fat</td><td>Ghana</td><td>20.67</td><td>1,197</td><td>1,197/0/0</td><td><code>fat_Latn</code></td><td>omnilingual</td></tr>
<tr><td>112</td><td>Moore</td><td>mos</td><td>Burkina Faso</td><td>20.51</td><td>6,785</td><td>6,293/196/296</td><td><code>moore_mos</code></td><td>afrispeech</td></tr>
<tr><td>113</td><td>Ahanta</td><td>aha</td><td>Ghana</td><td>20.30</td><td>6,908</td><td>6,038/441/429</td><td><code>ahanta_aha</code></td><td>afrispeech</td></tr>
<tr><td>114</td><td>Gusilay</td><td>gsl</td><td>Senegal</td><td>20.25</td><td>1,173</td><td>775/238/160</td><td><code>gsl_Latn</code></td><td>omnilingual</td></tr>
<tr><td>115</td><td>Rombo</td><td>rof</td><td>Kenya</td><td>20.20</td><td>1,170</td><td>653/241/276</td><td><code>rof_Latn</code></td><td>omnilingual</td></tr>
<tr><td>116</td><td>Douala</td><td>dua</td><td>Cameroon</td><td>20.12</td><td>6,521</td><td>5,832/145/544</td><td><code>douala_dua</code></td><td>afrispeech</td></tr>
<tr><td>117</td><td>Luo</td><td>luo</td><td>Kenya</td><td>19.66</td><td>6,829</td><td>5,959/367/503</td><td><code>luo_luo</code></td><td>afrispeech</td></tr>
<tr><td>118</td><td>Sepedi</td><td>nso</td><td>South Africa</td><td>19.59</td><td>7,016</td><td>6,435/396/185</td><td><code>sepedi_nso</code></td><td>afrispeech</td></tr>
<tr><td>119</td><td>Bissau Guinean Creole</td><td>pov</td><td>Guinea-Bissau</td><td>19.35</td><td>6,573</td><td>5,899/375/299</td><td><code>bissau_guinean_creole_pov</code></td><td>afrispeech</td></tr>
<tr><td>120</td><td>Ganda</td><td>lug</td><td>Uganda</td><td>19.27</td><td>6,757</td><td>5,455/664/638</td><td><code>lug_asr</code></td><td>waxal</td></tr>
<tr><td>121</td><td>Sudanese Arabic (Arab script)</td><td>apd</td><td>Central African Republic</td><td>18.94</td><td>1,097</td><td>816/156/125</td><td><code>apd_Arab</code></td><td>omnilingual</td></tr>
<tr><td>122</td><td>Nobiin</td><td>fia</td><td>Egypt</td><td>18.87</td><td>1,093</td><td>668/262/163</td><td><code>fia_Latn</code></td><td>omnilingual</td></tr>
<tr><td>123</td><td>Swahili (Congo)</td><td>swc</td><td>DR Congo</td><td>18.67</td><td>6,493</td><td>5,670/503/320</td><td><code>swahili_congo_swc</code></td><td>afrispeech</td></tr>
<tr><td>124</td><td>Western Maninkakan</td><td>mlq</td><td>Gambia</td><td>18.56</td><td>1,075</td><td>621/200/254</td><td><code>mlq_Latn</code></td><td>omnilingual</td></tr>
<tr><td>125</td><td>Judeo-Moroccan Arabic (Hebr script)</td><td>aju</td><td>Morocco</td><td>17.53</td><td>1,015</td><td>1,015/0/0</td><td><code>aju_Hebr</code></td><td>omnilingual</td></tr>
<tr><td>126</td><td>Sehwi</td><td>sfw</td><td>Ghana</td><td>17.48</td><td>6,024</td><td>5,488/199/337</td><td><code>sehwi_sfw</code></td><td>afrispeech</td></tr>
<tr><td>127</td><td>Runyankore</td><td>nyn</td><td>Uganda</td><td>17.10</td><td>5,605</td><td>4,736/432/437</td><td><code>runyankore_nyn</code></td><td>afrispeech</td></tr>
<tr><td>128</td><td>Boulou</td><td>bum</td><td>Cameroon</td><td>16.93</td><td>5,692</td><td>5,153/231/308</td><td><code>boulou_bum</code></td><td>afrispeech</td></tr>
<tr><td>129</td><td>Kwanyama</td><td>kua</td><td>Namibia</td><td>16.66</td><td>6,107</td><td>5,467/239/401</td><td><code>kwanyama_kua</code></td><td>afrispeech</td></tr>
<tr><td>130</td><td>Tooro</td><td>ttj</td><td>DR Congo</td><td>16.54</td><td>958</td><td>674/110/174</td><td><code>ttj_Latn</code></td><td>omnilingual</td></tr>
<tr><td>131</td><td>Yoruba</td><td>yor</td><td>Nigeria</td><td>16.10</td><td>2,611</td><td>2,233/201/177</td><td><code>yor_tts</code></td><td>waxal</td></tr>
<tr><td>132</td><td>Konzo</td><td>koo</td><td>DR Congo</td><td>15.94</td><td>923</td><td>654/146/123</td><td><code>koo_Latn</code></td><td>omnilingual</td></tr>
<tr><td>133</td><td>Jula</td><td>dyu</td><td>Burkina Faso</td><td>15.85</td><td>5,486</td><td>5,019/258/209</td><td><code>jula_dyu</code></td><td>afrispeech</td></tr>
<tr><td>134</td><td>Western Niger Fulfulde</td><td>fuh</td><td>Burkina Faso</td><td>15.64</td><td>906</td><td>906/0/0</td><td><code>fuh_Latn</code></td><td>omnilingual</td></tr>
<tr><td>135</td><td>Tandroy-Mahafaly Malagasy</td><td>tdx</td><td>Madagascar</td><td>15.54</td><td>900</td><td>450/150/300</td><td><code>tdx_Latn</code></td><td>omnilingual</td></tr>
<tr><td>136</td><td>Tshwa</td><td>tsc</td><td>Mozambique</td><td>14.99</td><td>5,221</td><td>4,651/209/361</td><td><code>tshwa_tsc</code></td><td>afrispeech</td></tr>
<tr><td>137</td><td>Acoli</td><td>ach</td><td>Uganda</td><td>14.70</td><td>5,155</td><td>4,108/519/528</td><td><code>ach_asr</code></td><td>waxal</td></tr>
<tr><td>138</td><td>Mandjak</td><td>mfv</td><td>Gambia</td><td>14.28</td><td>827</td><td>488/179/160</td><td><code>mfv_Latn</code></td><td>omnilingual</td></tr>
<tr><td>139</td><td>Yoruba</td><td>yor</td><td>Nigeria</td><td>14.19</td><td>4,960</td><td>4,504/216/240</td><td><code>yoruba_yor</code></td><td>afrispeech</td></tr>
<tr><td>140</td><td>Rangi</td><td>lag</td><td>Tanzania</td><td>14.16</td><td>820</td><td>477/180/163</td><td><code>lag_Latn</code></td><td>omnilingual</td></tr>
<tr><td>141</td><td>Balanta-Ganja</td><td>bjt</td><td>Guinea-Bissau</td><td>13.95</td><td>808</td><td>576/109/123</td><td><code>bjt_Latn</code></td><td>omnilingual</td></tr>
<tr><td>142</td><td>Chiga</td><td>cgg</td><td>DR Congo</td><td>13.90</td><td>805</td><td>559/126/120</td><td><code>cgg_Latn</code></td><td>omnilingual</td></tr>
<tr><td>143</td><td>Setswana</td><td>tsn</td><td>South Africa</td><td>13.84</td><td>4,828</td><td>4,232/344/252</td><td><code>setswana_tsn</code></td><td>afrispeech</td></tr>
<tr><td>144</td><td>Kikongo ya Leta</td><td>ktu</td><td>DR Congo</td><td>13.81</td><td>4,754</td><td>4,232/262/260</td><td><code>kikongo_ya_leta_ktu</code></td><td>afrispeech</td></tr>
<tr><td>145</td><td>Liberian English</td><td>lir</td><td>Liberia</td><td>13.76</td><td>797</td><td>650/71/76</td><td><code>lir_Latn</code></td><td>omnilingual</td></tr>
<tr><td>146</td><td>Gambian Wolof</td><td>wof</td><td>Gambia</td><td>13.21</td><td>765</td><td>548/116/101</td><td><code>wof_Latn</code></td><td>omnilingual</td></tr>
<tr><td>147</td><td>Chichewa</td><td>nya</td><td>Malawi</td><td>12.81</td><td>3,884</td><td>3,407/231/246</td><td><code>chichewa_nya</code></td><td>afrispeech</td></tr>
<tr><td>148</td><td>Akebu</td><td>keu</td><td>Togo</td><td>12.67</td><td>734</td><td>494/91/149</td><td><code>keu_Latn</code></td><td>omnilingual</td></tr>
<tr><td>149</td><td>Sango</td><td>sag</td><td>Central African Republic</td><td>12.60</td><td>4,193</td><td>3,645/261/287</td><td><code>sango_sag</code></td><td>afrispeech</td></tr>
<tr><td>150</td><td>Acoli</td><td>ach</td><td>Uganda</td><td>12.52</td><td>2,031</td><td>1,621/218/192</td><td><code>ach_tts</code></td><td>waxal</td></tr>
<tr><td>151</td><td>Kikuyu</td><td>kik</td><td>Kenya</td><td>12.49</td><td>2,026</td><td>1,602/210/214</td><td><code>kik_tts</code></td><td>waxal</td></tr>
<tr><td>152</td><td>Ganda</td><td>lug</td><td>Uganda</td><td>12.48</td><td>2,024</td><td>1,608/211/205</td><td><code>lug_tts</code></td><td>waxal</td></tr>
<tr><td>153</td><td>Mashi</td><td>shr</td><td>DR Congo</td><td>12.41</td><td>4,263</td><td>3,855/273/135</td><td><code>mashi_shr</code></td><td>afrispeech</td></tr>
<tr><td>154</td><td>Nigerian Pidgin</td><td>pcm</td><td>Nigeria</td><td>12.29</td><td>1,993</td><td>1,590/199/204</td><td><code>pcm_tts</code></td><td>waxal</td></tr>
<tr><td>155</td><td>Luo (Kenya and Tanzania)</td><td>luo</td><td>Kenya</td><td>12.26</td><td>1,989</td><td>1,552/228/209</td><td><code>luo_tts</code></td><td>waxal</td></tr>
<tr><td>156</td><td>Nyankole</td><td>nyn</td><td>Uganda</td><td>12.25</td><td>1,987</td><td>1,597/199/191</td><td><code>nyn_tts</code></td><td>waxal</td></tr>
<tr><td>157</td><td>Seychelles Creole</td><td>crs</td><td>Seychelles</td><td>12.16</td><td>4,269</td><td>3,983/72/214</td><td><code>seychelles_creole_crs</code></td><td>afrispeech</td></tr>
<tr><td>158</td><td>Hausa</td><td>hau</td><td>Nigeria</td><td>12.15</td><td>1,971</td><td>1,572/202/197</td><td><code>hau_tts</code></td><td>waxal</td></tr>
<tr><td>159</td><td>Bayot</td><td>bda</td><td>Gambia</td><td>11.83</td><td>685</td><td>450/158/77</td><td><code>bda_Latn</code></td><td>omnilingual</td></tr>
<tr><td>160</td><td>Igbo</td><td>ibo</td><td>Nigeria</td><td>11.78</td><td>1,911</td><td>1,552/159/200</td><td><code>ibo_tts</code></td><td>waxal</td></tr>
<tr><td>161</td><td>Fang</td><td>fan</td><td>Equatorial Guinea</td><td>11.68</td><td>3,969</td><td>3,630/184/155</td><td><code>fang_fan</code></td><td>afrispeech</td></tr>
<tr><td>162</td><td>Tugen</td><td>tuy</td><td>Kenya</td><td>11.52</td><td>667</td><td>407/104/156</td><td><code>tuy_Latn</code></td><td>omnilingual</td></tr>
<tr><td>163</td><td>Fipa</td><td>fip</td><td>Malawi</td><td>11.46</td><td>664</td><td>410/106/148</td><td><code>fip_Latn</code></td><td>omnilingual</td></tr>
<tr><td>164</td><td>Gweno</td><td>gwe</td><td>Kenya</td><td>11.34</td><td>657</td><td>401/123/133</td><td><code>gwe_Latn</code></td><td>omnilingual</td></tr>
<tr><td>165</td><td>Luganda</td><td>lug</td><td>Uganda</td><td>11.25</td><td>3,617</td><td>3,237/75/305</td><td><code>luganda_lug</code></td><td>afrispeech</td></tr>
<tr><td>166</td><td>Kituba</td><td>ktu</td><td>DR Congo</td><td>11.24</td><td>3,935</td><td>3,639/195/101</td><td><code>kituba_ktu</code></td><td>afrispeech</td></tr>
<tr><td>167</td><td>Tshiluba</td><td>lua</td><td>DR Congo</td><td>11.24</td><td>3,532</td><td>3,234/158/140</td><td><code>tshiluba_lua</code></td><td>afrispeech</td></tr>
<tr><td>168</td><td>Keiyo</td><td>eyo</td><td>Kenya</td><td>11.17</td><td>647</td><td>398/109/140</td><td><code>eyo_Latn</code></td><td>omnilingual</td></tr>
<tr><td>169</td><td>Farefare</td><td>gur</td><td>Burkina Faso</td><td>11.08</td><td>642</td><td>476/66/100</td><td><code>gur_Latn</code></td><td>omnilingual</td></tr>
<tr><td>170</td><td>Swahili (macrolanguage)</td><td>swa</td><td>Kenya</td><td>10.96</td><td>1,778</td><td>1,387/192/199</td><td><code>swa_tts</code></td><td>waxal</td></tr>
<tr><td>171</td><td>Chopi</td><td>cce</td><td>Mozambique</td><td>10.80</td><td>3,789</td><td>3,414/144/231</td><td><code>chopi_cce</code></td><td>afrispeech</td></tr>
<tr><td>172</td><td>Warji</td><td>wji</td><td>Nigeria</td><td>10.62</td><td>615</td><td>443/98/74</td><td><code>wji_Latn</code></td><td>omnilingual</td></tr>
<tr><td>173</td><td>Cibemba</td><td>bem</td><td>Zambia</td><td>10.56</td><td>3,823</td><td>3,333/170/320</td><td><code>cibemba_bem</code></td><td>afrispeech</td></tr>
<tr><td>174</td><td>Bago-Kusuntu</td><td>bqg</td><td>Togo</td><td>10.52</td><td>609</td><td>352/99/158</td><td><code>bqg_Latn</code></td><td>omnilingual</td></tr>
<tr><td>175</td><td>Anaang</td><td>anw</td><td>Nigeria</td><td>10.41</td><td>603</td><td>369/115/119</td><td><code>anw_Latn</code></td><td>omnilingual</td></tr>
<tr><td>176</td><td>Tahaggart Tamahaq (Tfng script)</td><td>thv</td><td>Algeria</td><td>10.29</td><td>596</td><td>298/149/149</td><td><code>thv_Tfng</code></td><td>omnilingual</td></tr>
<tr><td>177</td><td>Ngangela</td><td>nba</td><td>Angola</td><td>10.28</td><td>3,638</td><td>3,348/166/124</td><td><code>ngangela_nba</code></td><td>afrispeech</td></tr>
<tr><td>178</td><td>Gola</td><td>gol</td><td>Liberia</td><td>10.24</td><td>593</td><td>347/120/126</td><td><code>gol_Latn</code></td><td>omnilingual</td></tr>
<tr><td>179</td><td>Ndau (Western)</td><td>ndc</td><td>Mozambique</td><td>10.22</td><td>3,621</td><td>3,235/239/147</td><td><code>ndau_western_ndc</code></td><td>afrispeech</td></tr>
<tr><td>180</td><td>Gbari</td><td>gby</td><td>Nigeria</td><td>10.01</td><td>580</td><td>409/97/74</td><td><code>gby_Latn</code></td><td>omnilingual</td></tr>
<tr><td>181</td><td>Phimbi</td><td>phm</td><td>Mozambique</td><td>10.00</td><td>3,561</td><td>3,172/121/268</td><td><code>phimbi_phm</code></td><td>afrispeech</td></tr>
<tr><td>182</td><td>Gbagyi</td><td>gbr</td><td>Nigeria</td><td>9.82</td><td>569</td><td>396/94/79</td><td><code>gbr_Latn</code></td><td>omnilingual</td></tr>
<tr><td>183</td><td>Dũya</td><td>ldb</td><td>Nigeria</td><td>9.76</td><td>565</td><td>401/74/90</td><td><code>ldb_Latn</code></td><td>omnilingual</td></tr>
<tr><td>184</td><td>Dera (Nigeria)</td><td>kna</td><td>Nigeria</td><td>9.69</td><td>561</td><td>376/77/108</td><td><code>kna_Latn</code></td><td>omnilingual</td></tr>
<tr><td>185</td><td>Igo</td><td>ahl</td><td>Ghana</td><td>9.58</td><td>555</td><td>376/87/92</td><td><code>ahl_Latn</code></td><td>omnilingual</td></tr>
<tr><td>186</td><td>Mansoanka</td><td>msw</td><td>Gambia</td><td>9.58</td><td>555</td><td>379/88/88</td><td><code>msw_Latn</code></td><td>omnilingual</td></tr>
<tr><td>187</td><td>Samba Leko</td><td>ndi</td><td>Cameroon</td><td>9.58</td><td>555</td><td>349/134/72</td><td><code>ndi_Latn</code></td><td>omnilingual</td></tr>
<tr><td>188</td><td>Soninke</td><td>snk</td><td>Gambia</td><td>9.57</td><td>554</td><td>385/91/78</td><td><code>snk_Latn</code></td><td>omnilingual</td></tr>
<tr><td>189</td><td>Pero</td><td>pip</td><td>Nigeria</td><td>9.50</td><td>550</td><td>322/124/104</td><td><code>pip_Latn</code></td><td>omnilingual</td></tr>
<tr><td>190</td><td>Bondei</td><td>bou</td><td>Tanzania</td><td>9.43</td><td>546</td><td>373/103/70</td><td><code>bou_Latn</code></td><td>omnilingual</td></tr>
<tr><td>191</td><td>Guduf-Gava</td><td>gdf</td><td>Cameroon</td><td>9.43</td><td>546</td><td>376/102/68</td><td><code>gdf_Latn</code></td><td>omnilingual</td></tr>
<tr><td>192</td><td>Swati</td><td>ssw</td><td>South Africa</td><td>9.42</td><td>3,236</td><td>2,767/170/299</td><td><code>swati_ssw</code></td><td>afrispeech</td></tr>
<tr><td>193</td><td>Cakfem-Mushere</td><td>cky</td><td>Nigeria</td><td>9.41</td><td>545</td><td>307/123/115</td><td><code>cky_Latn</code></td><td>omnilingual</td></tr>
<tr><td>194</td><td>Ewe</td><td>ewe</td><td>Ghana</td><td>9.36</td><td>1,519</td><td>1,215/152/152</td><td><code>ewe_tts</code></td><td>waxal</td></tr>
<tr><td>195</td><td>Burak</td><td>bys</td><td>Nigeria</td><td>9.31</td><td>539</td><td>360/103/76</td><td><code>bys_Latn</code></td><td>omnilingual</td></tr>
<tr><td>196</td><td>Idoma</td><td>idu</td><td>Nigeria</td><td>9.24</td><td>535</td><td>348/115/72</td><td><code>idu_Latn</code></td><td>omnilingual</td></tr>
<tr><td>197</td><td>Kalabari</td><td>ijn</td><td>Nigeria</td><td>9.24</td><td>535</td><td>341/94/100</td><td><code>ijn_Latn</code></td><td>omnilingual</td></tr>
<tr><td>198</td><td>Vai</td><td>vai</td><td>Liberia</td><td>9.20</td><td>533</td><td>299/87/147</td><td><code>vai_Latn</code></td><td>omnilingual</td></tr>
<tr><td>199</td><td>Isekiri</td><td>its</td><td>Nigeria</td><td>9.19</td><td>532</td><td>369/80/83</td><td><code>its_Latn</code></td><td>omnilingual</td></tr>
<tr><td>200</td><td>Ngizim</td><td>ngi</td><td>Nigeria</td><td>9.13</td><td>529</td><td>352/104/73</td><td><code>ngi_Latn</code></td><td>omnilingual</td></tr>
<tr><td>201</td><td>Eastern Krahn</td><td>kqo</td><td>Côte d&#x27;Ivoire</td><td>9.12</td><td>528</td><td>280/106/142</td><td><code>kqo_Latn</code></td><td>omnilingual</td></tr>
<tr><td>202</td><td>Eloyi</td><td>afo</td><td>Nigeria</td><td>9.10</td><td>527</td><td>357/70/100</td><td><code>afo_Latn</code></td><td>omnilingual</td></tr>
<tr><td>203</td><td>Naba</td><td>mne</td><td>Chad</td><td>9.08</td><td>526</td><td>251/176/99</td><td><code>mne_Latn</code></td><td>omnilingual</td></tr>
<tr><td>204</td><td>Alago</td><td>ala</td><td>Nigeria</td><td>9.05</td><td>524</td><td>341/86/97</td><td><code>ala_Latn</code></td><td>omnilingual</td></tr>
<tr><td>205</td><td>Lijili</td><td>mgi</td><td>Nigeria</td><td>9.00</td><td>521</td><td>353/70/98</td><td><code>mgi_Latn</code></td><td>omnilingual</td></tr>
<tr><td>206</td><td>Ubaghara</td><td>byc</td><td>Nigeria</td><td>9.00</td><td>521</td><td>349/86/86</td><td><code>byc_Latn</code></td><td>omnilingual</td></tr>
<tr><td>207</td><td>Kinga</td><td>zga</td><td>Malawi</td><td>8.98</td><td>520</td><td>520/0/0</td><td><code>zga_Latn</code></td><td>omnilingual</td></tr>
<tr><td>208</td><td>Toupouri</td><td>tui</td><td>Cameroon</td><td>8.96</td><td>3,070</td><td>2,588/292/190</td><td><code>toupouri_tui</code></td><td>afrispeech</td></tr>
<tr><td>209</td><td>Abron</td><td>abr</td><td>Côte d&#x27;Ivoire</td><td>8.94</td><td>518</td><td>302/87/129</td><td><code>abr_Latn</code></td><td>omnilingual</td></tr>
<tr><td>210</td><td>Eleme</td><td>elm</td><td>Nigeria</td><td>8.91</td><td>516</td><td>350/90/76</td><td><code>elm_Latn</code></td><td>omnilingual</td></tr>
<tr><td>211</td><td>Mbe</td><td>mfo</td><td>Nigeria</td><td>8.87</td><td>514</td><td>341/80/93</td><td><code>mfo_Latn</code></td><td>omnilingual</td></tr>
<tr><td>212</td><td>Khana</td><td>ogo</td><td>Nigeria</td><td>8.79</td><td>509</td><td>332/77/100</td><td><code>ogo_Latn</code></td><td>omnilingual</td></tr>
<tr><td>213</td><td>Degema</td><td>deg</td><td>Nigeria</td><td>8.77</td><td>508</td><td>339/90/79</td><td><code>deg_Latn</code></td><td>omnilingual</td></tr>
<tr><td>214</td><td>Ejagham</td><td>etu</td><td>Cameroon</td><td>8.77</td><td>508</td><td>325/95/88</td><td><code>etu_Latn</code></td><td>omnilingual</td></tr>
<tr><td>215</td><td>Ikwere</td><td>ikw</td><td>Nigeria</td><td>8.70</td><td>504</td><td>318/91/95</td><td><code>ikw_Latn</code></td><td>omnilingual</td></tr>
<tr><td>216</td><td>Goemai</td><td>ank</td><td>Nigeria</td><td>8.68</td><td>503</td><td>325/107/71</td><td><code>ank_Latn</code></td><td>omnilingual</td></tr>
<tr><td>217</td><td>Isoko</td><td>iso</td><td>Nigeria</td><td>8.68</td><td>503</td><td>325/79/99</td><td><code>iso_Latn</code></td><td>omnilingual</td></tr>
<tr><td>218</td><td>Bura-Pabir</td><td>bwr</td><td>Nigeria</td><td>8.67</td><td>502</td><td>352/82/68</td><td><code>bwr_Latn</code></td><td>omnilingual</td></tr>
<tr><td>219</td><td>Venda</td><td>ven</td><td>South Africa</td><td>8.65</td><td>2,990</td><td>2,721/135/134</td><td><code>venda_ven</code></td><td>afrispeech</td></tr>
<tr><td>220</td><td>Basa (Nigeria)</td><td>bzw</td><td>Nigeria</td><td>8.63</td><td>500</td><td>336/72/92</td><td><code>bzw_Latn</code></td><td>omnilingual</td></tr>
<tr><td>221</td><td>Yace</td><td>ekr</td><td>Nigeria</td><td>8.63</td><td>500</td><td>351/74/75</td><td><code>ekr_Latn</code></td><td>omnilingual</td></tr>
<tr><td>222</td><td>Ito</td><td>itw</td><td>Nigeria</td><td>8.58</td><td>497</td><td>341/79/77</td><td><code>itw_Latn</code></td><td>omnilingual</td></tr>
<tr><td>223</td><td>Miya</td><td>mkf</td><td>Nigeria</td><td>8.58</td><td>497</td><td>342/67/88</td><td><code>mkf_Latn</code></td><td>omnilingual</td></tr>
<tr><td>224</td><td>Nigerian Fulfulde</td><td>fuv</td><td>Cameroon</td><td>8.56</td><td>496</td><td>347/74/75</td><td><code>fuv_Latn</code></td><td>omnilingual</td></tr>
<tr><td>225</td><td>Abua</td><td>abn</td><td>Nigeria</td><td>8.55</td><td>495</td><td>317/86/92</td><td><code>abn_Latn</code></td><td>omnilingual</td></tr>
<tr><td>226</td><td>Ngamo</td><td>nbh</td><td>Nigeria</td><td>8.53</td><td>494</td><td>311/73/110</td><td><code>nbh_Latn</code></td><td>omnilingual</td></tr>
<tr><td>227</td><td>Kuanyama</td><td>kua</td><td>Angola</td><td>8.48</td><td>491</td><td>312/95/84</td><td><code>kua_Latn</code></td><td>omnilingual</td></tr>
<tr><td>228</td><td>Kwambi</td><td>kwm</td><td>Namibia</td><td>8.48</td><td>491</td><td>328/75/88</td><td><code>kwm_Latn</code></td><td>omnilingual</td></tr>
<tr><td>229</td><td>Waja</td><td>wja</td><td>Nigeria</td><td>8.48</td><td>491</td><td>345/73/73</td><td><code>wja_Latn</code></td><td>omnilingual</td></tr>
<tr><td>230</td><td>Geji</td><td>gyz</td><td>Nigeria</td><td>8.46</td><td>490</td><td>315/92/83</td><td><code>gyz_Latn</code></td><td>omnilingual</td></tr>
<tr><td>231</td><td>Bade</td><td>bde</td><td>Nigeria</td><td>8.44</td><td>489</td><td>314/101/74</td><td><code>bde_Latn</code></td><td>omnilingual</td></tr>
<tr><td>232</td><td>Tsotso</td><td>lto</td><td>Kenya</td><td>8.44</td><td>489</td><td>316/98/75</td><td><code>lto_Latn</code></td><td>omnilingual</td></tr>
<tr><td>233</td><td>Logooli</td><td>rag</td><td>Kenya</td><td>8.43</td><td>488</td><td>306/72/110</td><td><code>rag_Latn</code></td><td>omnilingual</td></tr>
<tr><td>234</td><td>Dijim-Bwilim</td><td>cfa</td><td>Nigeria</td><td>8.41</td><td>487</td><td>334/69/84</td><td><code>cfa_Latn</code></td><td>omnilingual</td></tr>
<tr><td>235</td><td>Wapan</td><td>juk</td><td>Nigeria</td><td>8.41</td><td>487</td><td>307/75/105</td><td><code>juk_Latn</code></td><td>omnilingual</td></tr>
<tr><td>236</td><td>Boghom</td><td>bux</td><td>Nigeria</td><td>8.39</td><td>486</td><td>273/118/95</td><td><code>bux_Latn</code></td><td>omnilingual</td></tr>
<tr><td>237</td><td>Cibak</td><td>ckl</td><td>Nigeria</td><td>8.39</td><td>486</td><td>299/78/109</td><td><code>ckl_Latn</code></td><td>omnilingual</td></tr>
<tr><td>238</td><td>Kohumono</td><td>bcs</td><td>Nigeria</td><td>8.39</td><td>486</td><td>347/70/69</td><td><code>bcs_Latn</code></td><td>omnilingual</td></tr>
<tr><td>239</td><td>Mom Jango</td><td>ver</td><td>Cameroon</td><td>8.39</td><td>486</td><td>323/68/95</td><td><code>ver_Latn</code></td><td>omnilingual</td></tr>
<tr><td>240</td><td>Ashe</td><td>ahs</td><td>Nigeria</td><td>8.36</td><td>484</td><td>329/84/71</td><td><code>ahs_Latn</code></td><td>omnilingual</td></tr>
<tr><td>241</td><td>Nyungwe</td><td>nyu</td><td>Mozambique</td><td>8.35</td><td>2,976</td><td>2,744/63/169</td><td><code>nyungwe_nyu</code></td><td>afrispeech</td></tr>
<tr><td>242</td><td>Wanga</td><td>lwg</td><td>Kenya</td><td>8.34</td><td>483</td><td>297/76/110</td><td><code>lwg_Latn</code></td><td>omnilingual</td></tr>
<tr><td>243</td><td>Kimbundu</td><td>kmb</td><td>Angola</td><td>8.32</td><td>2,894</td><td>2,468/249/177</td><td><code>kimbundu_kmb</code></td><td>afrispeech</td></tr>
<tr><td>244</td><td>Mafa</td><td>maf</td><td>Cameroon</td><td>8.32</td><td>482</td><td>325/88/69</td><td><code>maf_Latn</code></td><td>omnilingual</td></tr>
<tr><td>245</td><td>Odual</td><td>odu</td><td>Nigeria</td><td>8.32</td><td>482</td><td>307/90/85</td><td><code>odu_Latn</code></td><td>omnilingual</td></tr>
<tr><td>246</td><td>Esan</td><td>ish</td><td>Nigeria</td><td>8.31</td><td>481</td><td>308/98/75</td><td><code>ish_Latn</code></td><td>omnilingual</td></tr>
<tr><td>247</td><td>Gusii</td><td>guz</td><td>Kenya</td><td>8.29</td><td>480</td><td>287/117/76</td><td><code>guz_Latn</code></td><td>omnilingual</td></tr>
<tr><td>248</td><td>Turkana</td><td>tuv</td><td>Ethiopia</td><td>8.27</td><td>479</td><td>285/106/88</td><td><code>tuv_Latn</code></td><td>omnilingual</td></tr>
<tr><td>249</td><td>Yekhee</td><td>ets</td><td>Nigeria</td><td>8.25</td><td>478</td><td>320/79/79</td><td><code>ets_Latn</code></td><td>omnilingual</td></tr>
<tr><td>250</td><td>Cen</td><td>cen</td><td>Nigeria</td><td>8.24</td><td>477</td><td>313/90/74</td><td><code>cen_Latn</code></td><td>omnilingual</td></tr>
<tr><td>251</td><td>Hwana</td><td>hwo</td><td>Nigeria</td><td>8.24</td><td>477</td><td>327/68/82</td><td><code>hwo_Latn</code></td><td>omnilingual</td></tr>
<tr><td>252</td><td>Koma</td><td>kmy</td><td>Cameroon</td><td>8.24</td><td>477</td><td>326/65/86</td><td><code>kmy_Latn</code></td><td>omnilingual</td></tr>
<tr><td>253</td><td>Ikposo</td><td>kpo</td><td>Ghana</td><td>8.22</td><td>476</td><td>293/90/93</td><td><code>kpo_Latn</code></td><td>omnilingual</td></tr>
<tr><td>254</td><td>Tangale</td><td>tan</td><td>Nigeria</td><td>8.22</td><td>476</td><td>323/69/84</td><td><code>tan_Latn</code></td><td>omnilingual</td></tr>
<tr><td>255</td><td>Karekare</td><td>kai</td><td>Nigeria</td><td>8.18</td><td>474</td><td>331/69/74</td><td><code>kai_Latn</code></td><td>omnilingual</td></tr>
<tr><td>256</td><td>Lamang</td><td>hia</td><td>Nigeria</td><td>8.18</td><td>474</td><td>324/85/65</td><td><code>hia_Latn</code></td><td>omnilingual</td></tr>
<tr><td>257</td><td>Cross River Mbembe</td><td>mfn</td><td>Nigeria</td><td>8.17</td><td>473</td><td>284/89/100</td><td><code>mfn_Latn</code></td><td>omnilingual</td></tr>
<tr><td>258</td><td>Idakho-Isukha-Tiriki</td><td>ida</td><td>Kenya</td><td>8.17</td><td>473</td><td>305/100/68</td><td><code>ida_Latn</code></td><td>omnilingual</td></tr>
<tr><td>259</td><td>Kinande</td><td>nnb</td><td>DR Congo</td><td>8.15</td><td>2,871</td><td>2,637/44/190</td><td><code>kinande_nnb</code></td><td>afrispeech</td></tr>
<tr><td>260</td><td>Huba</td><td>hbb</td><td>Nigeria</td><td>8.12</td><td>470</td><td>288/63/119</td><td><code>hbb_Latn</code></td><td>omnilingual</td></tr>
<tr><td>261</td><td>Marghi Central</td><td>mrt</td><td>Nigeria</td><td>8.12</td><td>470</td><td>297/80/93</td><td><code>mrt_Latn</code></td><td>omnilingual</td></tr>
<tr><td>262</td><td>Waci Gbe</td><td>wci</td><td>Benin</td><td>8.12</td><td>470</td><td>308/95/67</td><td><code>wci_Latn</code></td><td>omnilingual</td></tr>
<tr><td>263</td><td>Kamo</td><td>kcq</td><td>Nigeria</td><td>8.10</td><td>469</td><td>321/66/82</td><td><code>kcq_Latn</code></td><td>omnilingual</td></tr>
<tr><td>264</td><td>Nyankpa</td><td>yes</td><td>Nigeria</td><td>8.10</td><td>469</td><td>310/68/91</td><td><code>yes_Latn</code></td><td>omnilingual</td></tr>
<tr><td>265</td><td>Nsenga (Mozambique)</td><td>nse</td><td>Mozambique</td><td>8.09</td><td>2,865</td><td>2,385/313/167</td><td><code>nsenga_mozambique_nse</code></td><td>afrispeech</td></tr>
<tr><td>266</td><td>Baoule</td><td>bci</td><td>Côte d&#x27;Ivoire</td><td>8.08</td><td>2,657</td><td>2,307/125/225</td><td><code>baoule_bci</code></td><td>afrispeech</td></tr>
<tr><td>267</td><td>Bokyi</td><td>bky</td><td>Cameroon</td><td>8.08</td><td>468</td><td>297/93/78</td><td><code>bky_Latn</code></td><td>omnilingual</td></tr>
<tr><td>268</td><td>Jiba</td><td>juo</td><td>Nigeria</td><td>8.06</td><td>467</td><td>321/68/78</td><td><code>juo_Latn</code></td><td>omnilingual</td></tr>
<tr><td>269</td><td>Piya-Kwonci</td><td>piy</td><td>Nigeria</td><td>8.06</td><td>467</td><td>324/76/67</td><td><code>piy_Latn</code></td><td>omnilingual</td></tr>
<tr><td>270</td><td>Dadiya</td><td>dbd</td><td>Nigeria</td><td>8.05</td><td>466</td><td>289/70/107</td><td><code>dbd_Latn</code></td><td>omnilingual</td></tr>
<tr><td>271</td><td>Izon</td><td>ijc</td><td>Nigeria</td><td>8.05</td><td>466</td><td>290/80/96</td><td><code>ijc_Latn</code></td><td>omnilingual</td></tr>
<tr><td>272</td><td>Pökoot</td><td>pko</td><td>Kenya</td><td>8.03</td><td>465</td><td>298/70/97</td><td><code>pko_Latn</code></td><td>omnilingual</td></tr>
<tr><td>273</td><td>Lala-Roba</td><td>lla</td><td>Nigeria</td><td>7.99</td><td>463</td><td>301/91/71</td><td><code>lla_Latn</code></td><td>omnilingual</td></tr>
<tr><td>274</td><td>Gokana</td><td>gkn</td><td>Nigeria</td><td>7.98</td><td>2,808</td><td>2,567/179/62</td><td><code>gokana_gkn</code></td><td>afrispeech</td></tr>
<tr><td>275</td><td>Jju</td><td>kaj</td><td>Nigeria</td><td>7.96</td><td>461</td><td>272/91/98</td><td><code>kaj_Latn</code></td><td>omnilingual</td></tr>
<tr><td>276</td><td>Kabras</td><td>lkb</td><td>Kenya</td><td>7.96</td><td>461</td><td>284/72/105</td><td><code>lkb_Latn</code></td><td>omnilingual</td></tr>
<tr><td>277</td><td>Longuda</td><td>lnu</td><td>Nigeria</td><td>7.96</td><td>461</td><td>299/63/99</td><td><code>lnu_Latn</code></td><td>omnilingual</td></tr>
<tr><td>278</td><td>Nzanyi</td><td>nja</td><td>Cameroon</td><td>7.94</td><td>460</td><td>302/87/71</td><td><code>nja_Latn</code></td><td>omnilingual</td></tr>
<tr><td>279</td><td>Tarok</td><td>yer</td><td>Nigeria</td><td>7.94</td><td>460</td><td>298/70/92</td><td><code>yer_Latn</code></td><td>omnilingual</td></tr>
<tr><td>280</td><td>Buduma</td><td>bdm</td><td>Cameroon</td><td>7.93</td><td>459</td><td>301/90/68</td><td><code>bdm_Latn</code></td><td>omnilingual</td></tr>
<tr><td>281</td><td>Ngas</td><td>anc</td><td>Nigeria</td><td>7.91</td><td>458</td><td>314/65/79</td><td><code>anc_Latn</code></td><td>omnilingual</td></tr>
<tr><td>282</td><td>Kushi</td><td>kuh</td><td>Nigeria</td><td>7.89</td><td>457</td><td>308/77/72</td><td><code>kuh_Latn</code></td><td>omnilingual</td></tr>
<tr><td>283</td><td>Marghi South</td><td>mfm</td><td>Nigeria</td><td>7.89</td><td>457</td><td>315/78/64</td><td><code>mfm_Latn</code></td><td>omnilingual</td></tr>
<tr><td>284</td><td>Ndonga</td><td>ndo</td><td>Angola</td><td>7.89</td><td>457</td><td>285/68/104</td><td><code>ndo_Latn</code></td><td>omnilingual</td></tr>
<tr><td>285</td><td>Samba Daka</td><td>ccg</td><td>Cameroon</td><td>7.89</td><td>457</td><td>292/95/70</td><td><code>ccg_Latn</code></td><td>omnilingual</td></tr>
<tr><td>286</td><td>Saya</td><td>say</td><td>Nigeria</td><td>7.89</td><td>457</td><td>260/103/94</td><td><code>say_Latn</code></td><td>omnilingual</td></tr>
<tr><td>287</td><td>Eggon</td><td>ego</td><td>Nigeria</td><td>7.87</td><td>456</td><td>305/68/83</td><td><code>ego_Latn</code></td><td>omnilingual</td></tr>
<tr><td>288</td><td>Embu</td><td>ebu</td><td>Kenya</td><td>7.87</td><td>456</td><td>272/71/113</td><td><code>ebu_Latn</code></td><td>omnilingual</td></tr>
<tr><td>289</td><td>Glavda</td><td>glw</td><td>Cameroon</td><td>7.87</td><td>456</td><td>322/65/69</td><td><code>glw_Latn</code></td><td>omnilingual</td></tr>
<tr><td>290</td><td>Kanembu</td><td>kbl</td><td>Niger</td><td>7.87</td><td>456</td><td>306/79/71</td><td><code>kbl_Latn</code></td><td>omnilingual</td></tr>
<tr><td>291</td><td>Kirya-Konzəl</td><td>fkk</td><td>Nigeria</td><td>7.87</td><td>456</td><td>290/88/78</td><td><code>fkk_Latn</code></td><td>omnilingual</td></tr>
<tr><td>292</td><td>Kulung (Nigeria)</td><td>bbu</td><td>Nigeria</td><td>7.84</td><td>454</td><td>281/67/106</td><td><code>bbu_Latn</code></td><td>omnilingual</td></tr>
<tr><td>293</td><td>Meru</td><td>mer</td><td>Kenya</td><td>7.80</td><td>452</td><td>279/105/68</td><td><code>mer_Latn</code></td><td>omnilingual</td></tr>
<tr><td>294</td><td>Maba (Chad)</td><td>mde</td><td>Chad</td><td>7.68</td><td>445</td><td>445/0/0</td><td><code>mde_Latn</code></td><td>omnilingual</td></tr>
<tr><td>295</td><td>Tula</td><td>tul</td><td>Nigeria</td><td>7.68</td><td>445</td><td>263/112/70</td><td><code>tul_Latn</code></td><td>omnilingual</td></tr>
<tr><td>296</td><td>Afade</td><td>aal</td><td>Cameroon</td><td>7.67</td><td>444</td><td>310/64/70</td><td><code>aal_Latn</code></td><td>omnilingual</td></tr>
<tr><td>297</td><td>Bacama</td><td>bcy</td><td>Nigeria</td><td>7.65</td><td>443</td><td>278/99/66</td><td><code>bcy_Latn</code></td><td>omnilingual</td></tr>
<tr><td>298</td><td>Herero</td><td>her</td><td>Angola</td><td>7.65</td><td>443</td><td>281/95/67</td><td><code>her_Latn</code></td><td>omnilingual</td></tr>
<tr><td>299</td><td>Ndebele (Zimbabwe)</td><td>nde</td><td>Zimbabwe</td><td>7.62</td><td>2,552</td><td>2,308/81/163</td><td><code>ndebele_zimbabwe_nde</code></td><td>afrispeech</td></tr>
<tr><td>300</td><td>Zarma</td><td>dje</td><td>Burkina Faso</td><td>7.61</td><td>441</td><td>316/61/64</td><td><code>dje_Latn</code></td><td>omnilingual</td></tr>
<tr><td>301</td><td>Havu</td><td>hav</td><td>DR Congo</td><td>7.59</td><td>2,619</td><td>2,447/172/0</td><td><code>havu_hav</code></td><td>afrispeech</td></tr>
<tr><td>302</td><td>Bangwinji</td><td>bsj</td><td>Nigeria</td><td>7.58</td><td>439</td><td>286/84/69</td><td><code>bsj_Latn</code></td><td>omnilingual</td></tr>
<tr><td>303</td><td>Dghwede</td><td>dgh</td><td>Cameroon</td><td>7.58</td><td>439</td><td>263/109/67</td><td><code>dgh_Latn</code></td><td>omnilingual</td></tr>
<tr><td>304</td><td>Tedaga</td><td>tuq</td><td>Libya</td><td>7.56</td><td>438</td><td>287/87/64</td><td><code>tuq_Latn</code></td><td>omnilingual</td></tr>
<tr><td>305</td><td>Tera</td><td>ttr</td><td>Nigeria</td><td>7.53</td><td>436</td><td>294/75/67</td><td><code>ttr_Latn</code></td><td>omnilingual</td></tr>
<tr><td>306</td><td>Baoulé</td><td>bci</td><td>Côte d&#x27;Ivoire</td><td>7.50</td><td>1,216</td><td>972/122/122</td><td><code>bau_tts</code></td><td>waxal</td></tr>
<tr><td>307</td><td>Ibinda</td><td>yom</td><td>DR Congo</td><td>7.46</td><td>2,672</td><td>2,508/59/105</td><td><code>ibinda_yom</code></td><td>afrispeech</td></tr>
<tr><td>308</td><td>Dazaga</td><td>dzg</td><td>Libya</td><td>7.39</td><td>428</td><td>261/65/102</td><td><code>dzg_Latn</code></td><td>omnilingual</td></tr>
<tr><td>309</td><td>Awak</td><td>awo</td><td>Nigeria</td><td>7.37</td><td>427</td><td>265/71/91</td><td><code>awo_Latn</code></td><td>omnilingual</td></tr>
<tr><td>310</td><td>Fanti</td><td>fat</td><td>Ghana</td><td>7.22</td><td>1,171</td><td>953/117/101</td><td><code>fat_tts</code></td><td>waxal</td></tr>
<tr><td>311</td><td>Liberia Kpelle</td><td>xpe</td><td>Guinea</td><td>7.22</td><td>418</td><td>266/84/68</td><td><code>xpe_Latn</code></td><td>omnilingual</td></tr>
<tr><td>312</td><td>Agwagwune</td><td>yay</td><td>Nigeria</td><td>6.84</td><td>396</td><td>227/87/82</td><td><code>yay_Latn</code></td><td>omnilingual</td></tr>
<tr><td>313</td><td>Dinka</td><td>din</td><td>South Sudan</td><td>6.84</td><td>2,496</td><td>2,178/99/219</td><td><code>dinka_din</code></td><td>afrispeech</td></tr>
<tr><td>314</td><td>Itsekiri</td><td>its</td><td>Nigeria</td><td>6.78</td><td>2,387</td><td>2,387/0/0</td><td><code>itsekiri_its</code></td><td>afrispeech</td></tr>
<tr><td>315</td><td>Kwangali</td><td>kwn</td><td>Namibia</td><td>6.75</td><td>2,258</td><td>2,139/81/38</td><td><code>kwangali_kwn</code></td><td>afrispeech</td></tr>
<tr><td>316</td><td>Twi</td><td>twi</td><td>Ghana</td><td>6.74</td><td>1,093</td><td>872/117/104</td><td><code>twi_tts</code></td><td>waxal</td></tr>
<tr><td>317</td><td>Chitonga</td><td>toi</td><td>Zambia</td><td>6.61</td><td>2,487</td><td>2,220/195/72</td><td><code>chitonga_toi</code></td><td>afrispeech</td></tr>
<tr><td>318</td><td>Bassa (Liberia)</td><td>bsq</td><td>Liberia</td><td>6.07</td><td>2,284</td><td>2,135/12/137</td><td><code>bassa_liberia_bsq</code></td><td>afrispeech</td></tr>
<tr><td>319</td><td>Cinyanja</td><td>nya</td><td>Malawi</td><td>5.79</td><td>1,883</td><td>1,765/42/76</td><td><code>cinyanja_nya</code></td><td>afrispeech</td></tr>
<tr><td>320</td><td>Ndonga</td><td>ndo</td><td>Namibia</td><td>5.77</td><td>1,920</td><td>1,745/63/112</td><td><code>ndonga_ndo</code></td><td>afrispeech</td></tr>
<tr><td>321</td><td>Aja</td><td>ajg</td><td>Benin</td><td>5.71</td><td>1,942</td><td>1,765/105/72</td><td><code>aja_ajg</code></td><td>afrispeech</td></tr>
<tr><td>322</td><td>Kpelle</td><td>xpe</td><td>Liberia</td><td>5.70</td><td>1,933</td><td>1,743/112/78</td><td><code>kpelle_xpe</code></td><td>afrispeech</td></tr>
<tr><td>323</td><td>Réunion Creole</td><td>rcf</td><td>Réunion</td><td>5.25</td><td>1,803</td><td>1,718/23/62</td><td><code>r_union_creole_rcf</code></td><td>afrispeech</td></tr>
<tr><td>324</td><td>Eastern Egyptian Bedawi Arabic (Arab script)</td><td>avl</td><td>Egypt</td><td>5.18</td><td>300</td><td>300/0/0</td><td><code>avl_Arab</code></td><td>omnilingual</td></tr>
<tr><td>325</td><td>Algerian Saharan Arabic (Arab script)</td><td>aao</td><td>Algeria</td><td>5.15</td><td>298</td><td>298/0/0</td><td><code>aao_Arab</code></td><td>omnilingual</td></tr>
<tr><td>326</td><td>Ndebele</td><td>nbl</td><td>South Africa</td><td>5.15</td><td>1,985</td><td>1,819/134/32</td><td><code>ndebele_nbl</code></td><td>afrispeech</td></tr>
<tr><td>327</td><td>Chadian Arabic</td><td>shu</td><td>Cameroon</td><td>5.13</td><td>297</td><td>297/0/0</td><td><code>shu_Latn</code></td><td>omnilingual</td></tr>
<tr><td>328</td><td>Abbey</td><td>aba</td><td>Côte d&#x27;Ivoire</td><td>5.03</td><td>1,776</td><td>1,574/166/36</td><td><code>abbey_aba</code></td><td>afrispeech</td></tr>
<tr><td>329</td><td>Yombe</td><td>yom</td><td>DR Congo</td><td>4.88</td><td>1,634</td><td>1,390/149/95</td><td><code>yombe_yom</code></td><td>afrispeech</td></tr>
<tr><td>330</td><td>Kikongo</td><td>kwy</td><td>Angola</td><td>4.87</td><td>1,769</td><td>1,623/103/43</td><td><code>kikongo_kwy</code></td><td>afrispeech</td></tr>
<tr><td>331</td><td>Umbundu</td><td>umb</td><td>Angola</td><td>4.70</td><td>1,630</td><td>1,340/227/63</td><td><code>umbundu_umb</code></td><td>afrispeech</td></tr>
<tr><td>332</td><td>Chiyao</td><td>yao</td><td>Mozambique</td><td>4.61</td><td>1,647</td><td>1,423/103/121</td><td><code>chiyao_yao</code></td><td>afrispeech</td></tr>
<tr><td>333</td><td>Loma</td><td>lom</td><td>Liberia</td><td>4.52</td><td>1,540</td><td>1,350/41/149</td><td><code>loma_lom</code></td><td>afrispeech</td></tr>
<tr><td>334</td><td>Wolaita</td><td>wal</td><td>Ethiopia</td><td>4.28</td><td>1,476</td><td>1,250/133/93</td><td><code>wolaita_wal</code></td><td>afrispeech</td></tr>
<tr><td>335</td><td>Chitonga (Malawi)</td><td>tog</td><td>Malawi</td><td>4.16</td><td>1,480</td><td>1,363/65/52</td><td><code>chitonga_malawi_tog</code></td><td>afrispeech</td></tr>
<tr><td>336</td><td>Tiv</td><td>tiv</td><td>Nigeria</td><td>4.03</td><td>1,374</td><td>1,220/119/35</td><td><code>tiv_tiv</code></td><td>afrispeech</td></tr>
<tr><td>337</td><td>Lari</td><td>ldi</td><td>Congo</td><td>3.85</td><td>1,359</td><td>1,284/42/33</td><td><code>lari_ldi</code></td><td>afrispeech</td></tr>
<tr><td>338</td><td>Meru</td><td>mer</td><td>Kenya</td><td>3.83</td><td>1,277</td><td>1,139/138/0</td><td><code>meru_mer</code></td><td>afrispeech</td></tr>
<tr><td>339</td><td>Ewondo</td><td>ewo</td><td>Cameroon</td><td>3.71</td><td>1,305</td><td>1,134/32/139</td><td><code>ewondo_ewo</code></td><td>afrispeech</td></tr>
<tr><td>340</td><td>Kabyle</td><td>kab</td><td>Algeria</td><td>3.65</td><td>1,188</td><td>1,068/64/56</td><td><code>kabyle_kab</code></td><td>afrispeech</td></tr>
<tr><td>341</td><td>Khana</td><td>ogo</td><td>Nigeria</td><td>3.62</td><td>1,256</td><td>921/139/196</td><td><code>khana_ogo</code></td><td>afrispeech</td></tr>
<tr><td>342</td><td>Gitonga</td><td>toh</td><td>Mozambique</td><td>3.60</td><td>1,319</td><td>1,228/4/87</td><td><code>gitonga_toh</code></td><td>afrispeech</td></tr>
<tr><td>343</td><td>Tewe</td><td>twx</td><td>Mozambique</td><td>3.39</td><td>1,251</td><td>1,076/148/27</td><td><code>tewe_twx</code></td><td>afrispeech</td></tr>
<tr><td>344</td><td>Dangme</td><td>ada</td><td>Ghana</td><td>3.37</td><td>1,177</td><td>941/154/82</td><td><code>dangme_ada</code></td><td>afrispeech</td></tr>
<tr><td>345</td><td>Ndau</td><td>ndc</td><td>Mozambique</td><td>3.34</td><td>1,231</td><td>1,147/68/16</td><td><code>ndau_ndc</code></td><td>afrispeech</td></tr>
<tr><td>346</td><td>Guéré</td><td>gxx</td><td>Côte d&#x27;Ivoire</td><td>3.01</td><td>1,057</td><td>953/67/37</td><td><code>gu_r_gxx</code></td><td>afrispeech</td></tr>
<tr><td>347</td><td>Wolof</td><td>wol</td><td>Senegal</td><td>2.50</td><td>846</td><td>803/10/33</td><td><code>wolof_wol</code></td><td>afrispeech</td></tr>
<tr><td>348</td><td>Damara</td><td>naq</td><td>Namibia</td><td>2.46</td><td>789</td><td>698/62/29</td><td><code>damara_naq</code></td><td>afrispeech</td></tr>
<tr><td>349</td><td>Swahili (Katanga)</td><td>swc</td><td>DR Congo</td><td>2.40</td><td>866</td><td>798/32/36</td><td><code>swahili_katanga_swc</code></td><td>afrispeech</td></tr>
<tr><td>350</td><td>Yacouba</td><td>daf</td><td>Côte d&#x27;Ivoire</td><td>2.26</td><td>787</td><td>656/0/131</td><td><code>yacouba_daf</code></td><td>afrispeech</td></tr>
<tr><td>351</td><td>Manyawa</td><td>mny</td><td>Mozambique</td><td>1.94</td><td>697</td><td>697/0/0</td><td><code>manyawa_mny</code></td><td>afrispeech</td></tr>
<tr><td>352</td><td>Makhuwa-Marrevone</td><td>xmc</td><td>Mozambique</td><td>1.87</td><td>704</td><td>635/41/28</td><td><code>makhuwa_marrevone_xmc</code></td><td>afrispeech</td></tr>
<tr><td>353</td><td>Makhuwa-Meetto</td><td>mgh</td><td>Mozambique</td><td>1.83</td><td>671</td><td>552/85/34</td><td><code>makhuwa_meetto_mgh</code></td><td>afrispeech</td></tr>
<tr><td>354</td><td>Cinamwanga</td><td>mwn</td><td>Zambia</td><td>1.66</td><td>553</td><td>500/53/0</td><td><code>cinamwanga_mwn</code></td><td>afrispeech</td></tr>
<tr><td>355</td><td>Chitonga (Zimbabwe)</td><td>toi</td><td>Zimbabwe</td><td>1.41</td><td>486</td><td>452/0/34</td><td><code>chitonga_zimbabwe_toi</code></td><td>afrispeech</td></tr>
<tr><td>356</td><td>Attié</td><td>ati</td><td>Côte d&#x27;Ivoire</td><td>1.40</td><td>482</td><td>477/5/0</td><td><code>atti_ati</code></td><td>afrispeech</td></tr>
<tr><td>357</td><td>Lunda</td><td>lun</td><td>Zambia</td><td>1.11</td><td>423</td><td>322/101/0</td><td><code>lunda_lun</code></td><td>afrispeech</td></tr>
<tr><td>358</td><td>Lomwe</td><td>ngl</td><td>Mozambique</td><td>1.07</td><td>378</td><td>373/5/0</td><td><code>lomwe_ngl</code></td><td>afrispeech</td></tr>
<tr><td>359</td><td>Chuabo</td><td>chw</td><td>Mozambique</td><td>1.03</td><td>390</td><td>373/0/17</td><td><code>chuabo_chw</code></td><td>afrispeech</td></tr>
<tr><td>360</td><td>Mambwe-Lungu</td><td>mgr</td><td>Zambia</td><td>0.93</td><td>331</td><td>264/34/33</td><td><code>mambwe_lungu_mgr</code></td><td>afrispeech</td></tr>
<tr><td>361</td><td>Ijaw</td><td>ijc</td><td>Nigeria</td><td>0.88</td><td>312</td><td>292/0/20</td><td><code>ijaw_ijc</code></td><td>afrispeech</td></tr>
<tr><td>362</td><td>Ngbandi (Northern)</td><td>ngb</td><td>DR Congo</td><td>0.78</td><td>255</td><td>222/33/0</td><td><code>ngbandi_northern_ngb</code></td><td>afrispeech</td></tr>
<tr><td>363</td><td>Makhuwa-Shirima</td><td>vmk</td><td>Mozambique</td><td>0.77</td><td>277</td><td>277/0/0</td><td><code>makhuwa_shirima_vmk</code></td><td>afrispeech</td></tr>
<tr><td>364</td><td>Herero</td><td>her</td><td>Namibia</td><td>0.61</td><td>192</td><td>192/0/0</td><td><code>herero_her</code></td><td>afrispeech</td></tr>
<tr><td>365</td><td>Chokwe</td><td>cjk</td><td>Angola</td><td>0.56</td><td>177</td><td>169/0/8</td><td><code>chokwe_cjk</code></td><td>afrispeech</td></tr>
<tr><td>366</td><td>Taabwa</td><td>tap</td><td>DR Congo</td><td>0.56</td><td>195</td><td>163/0/32</td><td><code>taabwa_tap</code></td><td>afrispeech</td></tr>
<tr><td>367</td><td>Kisonge</td><td>sop</td><td>DR Congo</td><td>0.40</td><td>141</td><td>141/0/0</td><td><code>kisonge_sop</code></td><td>afrispeech</td></tr>
<tr><td>368</td><td>Kanyok</td><td>kny</td><td>DR Congo</td><td>0.28</td><td>109</td><td>109/0/0</td><td><code>kanyok_kny</code></td><td>afrispeech</td></tr>
<tr><td>369</td><td>Luvale</td><td>lue</td><td>Zambia</td><td>0.07</td><td>20</td><td>20/0/0</td><td><code>luvale_lue</code></td><td>afrispeech</td></tr>
</tbody>
</table>

</details>

## License

CC-BY-4.0
