"""Pull selected samples from the source dataset and standardise them.

The output is a :class:`datasets.Dataset` (or :class:`~datasets.DatasetDict`)
with a fixed, training-ready schema regardless of how the source subsets are
laid out:

    audio     -> datasets.Audio (decoded array + sampling_rate)
    text      -> transcription string
    language  -> human language label (from the catalog)
    country   -> ISO 3166-1 alpha-2 code (from the catalog)
    length    -> clip duration in seconds (float)

It also keeps ``subset`` and ``iso`` so a built dataset is self-describing.

The source subsets have columns:
    id, audio, text, duration, source, jw_code, iso639_3
Only ``audio`` is carried through verbatim; ``text`` and ``duration`` are
remapped to ``text``/``length``. Everything else (``id``, ``source``,
``jw_code``, ``iso639_3``) is intentionally dropped — ``iso`` is repopulated
from the catalog instead.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

from .catalog import DATASET_ID, LanguageEntry, by_subset

_DIGIT_RE = re.compile(r"\d")


def _text_has_numbers(text) -> bool:
    """Return True if *text* contains any digit character (0-9).

    Clips whose transcription contains digits are excluded by default because
    number-heavy transcripts tend to have poor text–audio alignment.
    """
    return bool(text and _DIGIT_RE.search(text))

STANDARD_COLUMNS = ["audio", "text", "language", "country", "length", "iso", "subset"]

# Candidate source column names we map onto our standard `text` column.
_TEXT_CANDIDATES = (
    "text", "sentence", "transcript", "transcription", "transcription_text",
    "normalized_text", "normalised_text", "translation",
)
# Candidate source column names that already carry a duration in seconds.
_DURATION_CANDIDATES = ("length", "duration", "duration_seconds", "audio_length")


def _pick_text_column(columns: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in _TEXT_CANDIDATES:
        if cand in lower:
            return lower[cand]
    return None


def _pick_duration_column(columns: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in _DURATION_CANDIDATES:
        if cand in lower:
            return lower[cand]
    return None


def build_subset(
    entry: LanguageEntry,
    *,
    split: str = "train",
    per_language: int | None = None,
    max_seconds: float | None = None,
    min_clip_seconds: float | None = None,
    max_clip_seconds: float | None = None,
    seed: int = 42,
    dataset_id: str = DATASET_ID,
    token: str | None = None,
    streaming: bool = False,
    on_clip=None,
):
    """Load one subset, cap it, and reshape to the standard schema.

    Returns a :class:`datasets.Dataset`. ``split="all"`` concatenates the
    available splits. Sampling (when ``per_language`` caps the size) is a
    deterministic shuffle so reruns with the same seed are reproducible.

    Per-clip length filter applied first (clips outside the range are dropped
    and never counted toward the caps):
      * ``min_clip_seconds`` / ``max_clip_seconds`` — keep clips whose duration
        is within this range.

    Two independent caps then bound the pull (whichever is hit first stops it):
      * ``per_language`` — max number of clips.
      * ``max_seconds`` — duration budget: accumulate clips until their summed
        ``length`` is as close as possible to this many seconds.

    When ``streaming`` is True the subset is streamed and only the needed
    examples are pulled (audio bytes are not decoded). This transfers almost
    nothing for a capped request — essential on a small Space where a full
    parquet download would time out.
    """
    if streaming:
        return _build_subset_streaming(
            entry, split=split, per_language=per_language, max_seconds=max_seconds,
            min_clip_seconds=min_clip_seconds, max_clip_seconds=max_clip_seconds,
            seed=seed, dataset_id=dataset_id, token=token, on_clip=on_clip,
        )

    from datasets import Audio, concatenate_datasets, load_dataset

    def _load(sp: str):
        return load_dataset(dataset_id, entry.subset, split=sp, token=token)

    if split == "all":
        parts = []
        for sp in ("train", "val", "test"):
            if entry.split_size(sp) > 0:
                parts.append(_load(sp))
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    else:
        ds = _load(split)

    # Drop out-of-range clips up front (uses the source duration column).
    _dc = _pick_duration_column(ds.column_names)
    if _dc and (min_clip_seconds is not None or max_clip_seconds is not None):
        lo = min_clip_seconds if min_clip_seconds is not None else float("-inf")
        hi = max_clip_seconds if max_clip_seconds is not None else float("inf")
        ds = ds.filter(
            lambda b: [(x is not None and lo <= float(x) <= hi) for x in b[_dc]],
            batched=True, desc=f"length filter {entry.subset}",
        )

    # Drop clips whose transcription contains digits — number-heavy text
    # correlates with poor text–audio alignment.
    _tc = _pick_text_column(ds.column_names)
    if _tc:
        ds = ds.filter(
            lambda b: [not _text_has_numbers(t) for t in b[_tc]],
            batched=True, desc=f"number filter {entry.subset}",
        )

    if (per_language is not None or max_seconds is not None) and len(ds):
        ds = ds.shuffle(seed=seed)
    if per_language is not None and len(ds) > per_language:
        ds = ds.select(range(per_language))

    src_cols = ds.column_names
    text_col = _pick_text_column(src_cols)
    dur_col = _pick_duration_column(src_cols)

    if "audio" not in src_cols:
        raise ValueError(
            f"Subset '{entry.subset}' has no 'audio' column (found {src_cols})."
        )

    # Reference a non-audio column for batch size so we don't force-decode audio
    # on the common path (duration is read from the metadata column).
    ref_col = dur_col or text_col or "audio"

    def _standardise(batch):
        n = len(batch[ref_col])
        out = {
            "text": list(batch[text_col]) if text_col else [""] * n,
            "language": [entry.language] * n,
            "country": [entry.country] * n,
            "iso": [entry.iso] * n,
            "subset": [entry.subset] * n,
        }
        if dur_col:
            out["length"] = [float(x) if x is not None else 0.0 for x in batch[dur_col]]
        else:
            # No duration column: decode audio to measure it.
            out["length"] = [_audio_len(a) for a in batch["audio"]]
        return out

    keep = {"audio"}
    remove = [c for c in ds.column_names if c not in keep]
    ds = ds.map(_standardise, batched=True, remove_columns=remove,
                desc=f"standardising {entry.subset}")
    # Apply the duration budget on the standardised `length` column.
    if max_seconds is not None and len(ds):
        k = _duration_cutoff(ds["length"], max_seconds)
        ds = ds.select(range(k))
    # Keep audio as a decodable Audio feature in the output.
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio())
    # Order columns predictably.
    ds = ds.select_columns([c for c in STANDARD_COLUMNS if c in ds.column_names])
    return ds


def _duration_cutoff(lengths, target: float) -> int:
    """Number of leading clips whose summed length is closest to ``target``.

    Accumulates until the sum reaches ``target``; if including the crossing clip
    overshoots further than stopping just before it, that last clip is dropped.
    """
    acc = 0.0
    k = 0
    for L in lengths:
        if acc >= target:
            break
        acc += float(L or 0.0)
        k += 1
    if k > 1 and acc > target:
        prev = acc - float(lengths[k - 1] or 0.0)
        if abs(prev - target) < abs(acc - target):
            k -= 1
    return k


def _build_subset_streaming(
    entry: LanguageEntry,
    *,
    split: str,
    per_language: int | None,
    max_seconds: float | None,
    min_clip_seconds: float | None = None,
    max_clip_seconds: float | None = None,
    seed: int,
    dataset_id: str,
    token: str | None,
    on_clip=None,
):
    """Stream a subset and pull only what's needed (no full download).

    Honours both ``per_language`` (clip count) and ``max_seconds`` (duration
    budget), stopping at whichever is reached first. Audio is kept undecoded
    ({bytes, path}) for speed/memory, then materialised into a Dataset with a
    proper Audio feature so the output is still playable and training-ready.
    """
    from datasets import Audio, Dataset, Features, Value, load_dataset

    splits = ("train", "validation", "test") if split == "all" else (split,)
    # Map our catalog's "val" terminology to the Hub's "validation" split name.
    split_alias = {"val": "validation"}

    # Size the shuffle buffer from whichever cap is set.
    avg = (entry.hours * 3600 / entry.clips) if entry.clips else 1.0
    est = per_language or (int(max_seconds / avg) + 1 if max_seconds else 0)
    buffer = max(1000, min(10000, est * 4)) if est else 1000

    def _stream_one(sp: str):
        ds = load_dataset(dataset_id, entry.subset, split=split_alias.get(sp, sp),
                          streaming=True, token=token)
        if per_language is not None or max_seconds is not None:
            ds = ds.shuffle(seed=seed, buffer_size=buffer)
        return ds.cast_column("audio", Audio(decode=False))

    rows: list[dict] = []
    acc = 0.0
    for sp in splits:
        stream = _stream_one(sp)
        cols = list(stream.features) if stream.features else []
        text_col = _pick_text_column(cols)
        dur_col = _pick_duration_column(cols)
        stop = False
        for ex in stream:
            if per_language is not None and len(rows) >= per_language:
                stop = True
                break
            if max_seconds is not None and acc >= max_seconds:
                stop = True
                break
            audio = ex.get("audio")  # {"bytes":..., "path":...} (undecoded)
            length = float(ex[dur_col]) if dur_col and ex.get(dur_col) is not None else _audio_len(audio)
            # Drop clips outside the requested length range (not counted).
            if min_clip_seconds is not None and length < min_clip_seconds:
                continue
            if max_clip_seconds is not None and length > max_clip_seconds:
                continue
            # Drop clips whose transcription contains digits.
            clip_text = ex.get(text_col, "") if text_col else ""
            if _text_has_numbers(clip_text):
                continue
            rows.append({
                "audio": audio,
                "text": clip_text,
                "language": entry.language,
                "country": entry.country,
                "length": length,
                "iso": entry.iso,
                "subset": entry.subset,
            })
            acc += length
            if on_clip:
                on_clip()
        if stop:
            break

    # Closeness trim: drop the crossing clip if stopping short is nearer the target.
    if max_seconds is not None and len(rows) > 1 and acc > max_seconds:
        if abs((acc - rows[-1]["length"]) - max_seconds) < abs(acc - max_seconds):
            rows.pop()

    features = Features({
        "audio": Audio(),
        "text": Value("string"),
        "language": Value("string"),
        "country": Value("string"),
        "length": Value("float64"),
        "iso": Value("string"),
        "subset": Value("string"),
    })
    return Dataset.from_list(rows, features=features)


def _audio_len(audio) -> float:
    """Duration in seconds from a decoded HF audio value.

    Handles both the classic dict form ({"array", "sampling_rate"}) and the
    newer torchcodec ``AudioDecoder`` returned by datasets>=4.
    """
    if audio is None:
        return 0.0
    # Classic dict form.
    if isinstance(audio, dict):
        arr = audio.get("array")
        sr = audio.get("sampling_rate")
        if arr is not None and sr:
            return round(len(arr) / sr, 3)
        return 0.0
    # torchcodec AudioDecoder: prefer cheap header metadata, else decode.
    meta = getattr(audio, "metadata", None)
    dur = getattr(meta, "duration_seconds", None) if meta is not None else None
    if dur:
        return round(float(dur), 3)
    try:
        return round(float(audio.get_all_samples().duration_seconds), 3)
    except Exception:
        return 0.0


def build_dataset(
    entries: Iterable[LanguageEntry],
    *,
    split: str = "train",
    per_language: int | None = None,
    max_seconds: float | None = None,
    min_clip_seconds: float | None = None,
    max_clip_seconds: float | None = None,
    target_sampling_rate: int | None = None,
    schema: str | None = None,
    seed: int = 42,
    dataset_id: str = DATASET_ID,
    token: str | None = None,
    streaming: bool = False,
    progress: Callable[[str], None] | None = None,
    on_clip=None,
):
    """Build a single combined :class:`datasets.Dataset` for the selection.

    Materialises the selection in memory (use this when you genuinely need a
    local copy — e.g. an offline working set). For training, prefer
    :func:`stream_dataset`, which streams lazily and never writes a copy.

    ``entries`` may be :class:`LanguageEntry` objects or subset-name strings.
    ``per_language`` caps clips per language; ``max_seconds`` caps total audio
    duration per language (whichever is hit first). ``min_clip_seconds`` /
    ``max_clip_seconds`` drop individual clips outside that length range before
    counting. ``target_sampling_rate`` resamples audio; ``schema`` reshapes
    columns for a training framework (see :func:`apply_schema`).
    """
    from datasets import concatenate_datasets

    resolved: list[LanguageEntry] = []
    for e in entries:
        resolved.append(e if isinstance(e, LanguageEntry) else _require(by_subset(e), e))

    parts = []
    for i, entry in enumerate(resolved, 1):
        if progress:
            progress(f"[{i}/{len(resolved)}] pulling {entry.language} ({entry.subset})…")
        parts.append(
            build_subset(
                entry, split=split, per_language=per_language, max_seconds=max_seconds,
                min_clip_seconds=min_clip_seconds, max_clip_seconds=max_clip_seconds,
                seed=seed, dataset_id=dataset_id, token=token, streaming=streaming,
                on_clip=on_clip,
            )
        )
    if not parts:
        raise ValueError("No languages selected — nothing to build.")
    combined = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    combined = apply_schema(combined, target_sampling_rate, schema)
    if progress:
        progress(f"Done: {len(combined)} clips across {len(parts)} languages.")
    return combined


def _require(entry: LanguageEntry | None, name: str) -> LanguageEntry:
    if entry is None:
        raise KeyError(f"Unknown subset '{name}'.")
    return entry


# Column shapes expected by common training pipelines.
SCHEMAS = {
    None: None,                                   # standard schema, unchanged
    "default": None,
    "asr": (None, ["audio", "text"]),             # minimal: just audio + transcript
    "whisper": ({"text": "sentence"}, ["audio", "sentence"]),
    "common_voice": ({"text": "sentence"}, ["audio", "sentence", "language", "country", "length"]),
}


def apply_schema(ds, target_sampling_rate: int | None = None, schema: str | None = None):
    """Resample audio and/or reshape columns for a training framework.

    Works on both :class:`datasets.Dataset` and :class:`datasets.IterableDataset`.
    ``schema`` is one of :data:`SCHEMAS`: ``asr`` (audio+text), ``whisper`` /
    ``common_voice`` (text→sentence), or ``None``/``default`` (unchanged).
    """
    from datasets import Audio

    if target_sampling_rate:
        ds = ds.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
    if schema not in SCHEMAS:
        raise ValueError(f"Unknown schema '{schema}'. Options: {sorted(k for k in SCHEMAS if k)}")
    spec = SCHEMAS.get(schema)
    if spec:
        rename, keep = spec
        if rename:
            for old, new in rename.items():
                if old in ds.column_names:
                    ds = ds.rename_column(old, new)
        ds = ds.select_columns([c for c in keep if c in ds.column_names])
    return ds


def _iter_rows(subsets, split, per_language, max_seconds,
               min_clip_seconds, max_clip_seconds, seed, dataset_id, token):
    """Generator yielding standardised rows (audio undecoded) for a selection.

    Lazily streams each subset and applies the per-language caps and per-clip
    length filter. Used by :func:`stream_dataset` to feed a training loop without
    materialising or saving any copy of the data.
    """
    from datasets import Audio, load_dataset

    split_alias = {"val": "validation"}
    splits = ("train", "validation", "test") if split == "all" else (split,)
    for sub in subsets:
        entry = by_subset(sub)
        if entry is None:
            continue
        avg = (entry.hours * 3600 / entry.clips) if entry.clips else 1.0
        est = per_language or (int(max_seconds / avg) + 1 if max_seconds else 0)
        buffer = max(1000, min(10000, est * 4)) if est else 1000
        count, acc, stop = 0, 0.0, False
        for sp in splits:
            ds = load_dataset(dataset_id, sub, split=split_alias.get(sp, sp),
                              streaming=True, token=token)
            if per_language is not None or max_seconds is not None:
                ds = ds.shuffle(seed=seed, buffer_size=buffer)
            ds = ds.cast_column("audio", Audio(decode=False))
            cols = list(ds.features) if ds.features else []
            text_col = _pick_text_column(cols)
            dur_col = _pick_duration_column(cols)
            for ex in ds:
                if per_language is not None and count >= per_language:
                    stop = True
                    break
                if max_seconds is not None and acc >= max_seconds:
                    stop = True
                    break
                audio = ex.get("audio")
                length = float(ex[dur_col]) if dur_col and ex.get(dur_col) is not None else _audio_len(audio)
                if min_clip_seconds is not None and length < min_clip_seconds:
                    continue
                if max_clip_seconds is not None and length > max_clip_seconds:
                    continue
                # Drop clips whose transcription contains digits.
                clip_text = ex.get(text_col, "") if text_col else ""
                if _text_has_numbers(clip_text):
                    continue
                yield {
                    "audio": audio,
                    "text": clip_text,
                    "language": entry.language, "country": entry.country,
                    "length": length, "iso": entry.iso, "subset": entry.subset,
                }
                count += 1
                acc += length
            if stop:
                break


def stream_dataset(
    entries,
    *,
    split: str = "train",
    per_language: int | None = None,
    max_seconds: float | None = None,
    min_clip_seconds: float | None = None,
    max_clip_seconds: float | None = None,
    target_sampling_rate: int | None = None,
    schema: str | None = None,
    seed: int = 42,
    dataset_id: str = DATASET_ID,
    token: str | None = None,
):
    """Stream a selection straight into training as a lazy ``IterableDataset``.

    This is the recommended path: it pulls samples on the fly and **never writes
    a local copy** of the dataset — you select languages and feed the result
    directly to your trainer. Same selection knobs as :func:`build_dataset`
    (caps, duration budget, length window), plus ``target_sampling_rate`` and
    ``schema`` to match your training framework. Returns a
    :class:`datasets.IterableDataset` with the standard schema (or the reshaped
    one). The duration budget may overshoot by at most one clip in this mode.
    """
    from datasets import Audio, Features, IterableDataset, Value

    subsets = [e.subset if isinstance(e, LanguageEntry) else e for e in entries]
    feats = Features({
        "audio": Audio(), "text": Value("string"), "language": Value("string"),
        "country": Value("string"), "length": Value("float64"),
        "iso": Value("string"), "subset": Value("string"),
    })
    ds = IterableDataset.from_generator(
        _iter_rows, features=feats,
        gen_kwargs=dict(
            subsets=subsets, split=split, per_language=per_language, max_seconds=max_seconds,
            min_clip_seconds=min_clip_seconds, max_clip_seconds=max_clip_seconds,
            seed=seed, dataset_id=dataset_id, token=token,
        ),
    )
    return apply_schema(ds, target_sampling_rate, schema)
