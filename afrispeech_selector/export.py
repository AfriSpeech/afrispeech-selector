"""Export a selection: an HF archive, a push to the Hub, or a TTS-ready dataset.

The TTS exporters write a local *training working set* (WAVs + a metadata
manifest) in the layout a given framework's data-prep expects. This is a working
copy for your own training, not for redistribution — mind the source dataset's
license before sharing.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

# TTS layouts the exporter can produce.
TTS_FORMATS = ("ljspeech", "piper", "vits", "melo")


def export_archive(dataset, out_dir: str | Path | None = None, name: str = "afrispeech_selection"):
    """Save the dataset to disk (load_from_disk format) and zip it.

    Returns the path to the ``.zip``. The archive round-trips with
    ``datasets.load_from_disk`` after unzipping, so audio stays decoded and
    training-ready.
    """
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="afrispeech_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = out_dir / name
    dataset.save_to_disk(str(ds_dir))
    zip_base = str(out_dir / name)
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=str(ds_dir))
    return zip_path


def export_parquet(dataset, out_dir: str | Path | None = None, name: str = "afrispeech_selection"):
    """Write the dataset to a single Parquet file (audio embedded as bytes).

    Returns the path to the ``.parquet``. Good for a one-file download that
    loads with ``datasets.Dataset.from_parquet`` or pandas/pyarrow.
    """
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="afrispeech_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    dataset.to_parquet(str(path))
    return str(path)


def export_metadata_csv(dataset, out_dir: str | Path | None = None, name: str = "afrispeech_selection"):
    """Write a manifest CSV of everything except the audio bytes."""
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="afrispeech_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_manifest.csv"
    cols = [c for c in dataset.column_names if c != "audio"]
    dataset.remove_columns([c for c in dataset.column_names if c not in cols]).to_csv(str(path))
    return str(path)


def push_to_hub(
    dataset,
    repo_id: str,
    *,
    token: str,
    private: bool = True,
    split: str = "train",
):
    """Push the built dataset to a user-owned HF dataset repo.

    The repo is created if it does not exist. ``split`` names the split the
    selection is uploaded as. Returns the repo URL.
    """
    if not token:
        raise ValueError("A Hugging Face token with write access is required to push.")
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id must look like 'username/dataset-name'.")

    from datasets import DatasetDict

    payload = dataset if hasattr(dataset, "keys") else DatasetDict({split: dataset})
    payload.push_to_hub(repo_id, token=token, private=private)
    return f"https://huggingface.co/datasets/{repo_id}"


# --------------------------------------------------------------------------- #
# TTS training-data-prep exporters
# --------------------------------------------------------------------------- #
def _audio_array(au, target_sr):
    """Return (float32 mono array, sr) from a decoded HF audio value, resampling
    to ``target_sr`` if needed. Handles dict and torchcodec AudioDecoder forms."""
    import numpy as np

    if isinstance(au, dict):
        arr = np.asarray(au["array"], dtype="float32")
        sr = int(au["sampling_rate"])
    else:  # torchcodec AudioDecoder
        s = au.get_all_samples()
        arr = s.data.numpy().astype("float32").squeeze()
        sr = int(s.sample_rate)
    if arr.ndim > 1:  # stereo -> mono
        arr = arr.mean(axis=0)
    if target_sr and sr != target_sr:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return arr, sr


def export_tts(dataset, out_dir: str | Path | None = None, *, name: str = "afrispeech_tts",
               fmt: str = "ljspeech", sampling_rate: int = 22050):
    """Write a TTS training-data-prep dataset: WAVs + a framework-specific manifest.

    Layout (under ``<out_dir>/<name>/``):
      ``wavs/<id>.wav``         16-bit PCM mono at ``sampling_rate``
      plus one manifest, by ``fmt``:
        * ``ljspeech`` — ``metadata.csv``:  ``id|text|text``  (generic; Coqui/Tacotron)
        * ``piper``    — ``metadata.csv``:  ``id|speaker|text``  (Piper multi-speaker)
        * ``vits``     — ``filelist.txt``:  ``wavs/<id>.wav|<sid>|text`` + ``speakers.txt``
        * ``melo``     — ``metadata.list``: ``wavs/<id>.wav|speaker|LANG|text``  (MeloTTS)

    Speaker = language label, language code = ISO 639-3 — natural for the
    multilingual selections this tool produces. Returns the dataset directory.
    """
    import soundfile as sf
    from datasets import Audio

    if fmt not in TTS_FORMATS:
        raise ValueError(f"Unknown TTS format '{fmt}'. Options: {list(TTS_FORMATS)}")

    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="afrispeech_tts_"))
    base = out_dir / name
    wavs = base / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)

    ds = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    rows = []          # (uid, text, speaker, iso)
    speakers: dict[str, int] = {}
    for i, ex in enumerate(ds):
        arr, sr = _audio_array(ex["audio"], sampling_rate)
        uid = f"{ex.get('subset', 'clip')}_{i:06d}"
        sf.write(str(wavs / f"{uid}.wav"), arr, sr, subtype="PCM_16")
        # Transcript is written verbatim — no normalisation/cleaning (that's the
        # TTS framework's job). Only line-breaks/tabs are turned into spaces so
        # each record stays on a single manifest line.
        text = (ex.get("text") or "").replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()
        spk = ex.get("language") or ex.get("subset") or "spk"
        iso = (ex.get("iso") or "").upper()
        speakers.setdefault(spk, len(speakers))
        rows.append((uid, text, spk, iso))

    if fmt == "ljspeech":
        _write_lines(base / "metadata.csv", [f"{u}|{t}|{t}" for u, t, _, _ in rows])
    elif fmt == "piper":
        _write_lines(base / "metadata.csv", [f"{u}|{s}|{t}" for u, t, s, _ in rows])
    elif fmt == "vits":
        _write_lines(base / "filelist.txt",
                     [f"wavs/{u}.wav|{speakers[s]}|{t}" for u, t, s, _ in rows])
        _write_lines(base / "speakers.txt",
                     [f"{sid}\t{name_}" for name_, sid in sorted(speakers.items(), key=lambda kv: kv[1])])
    elif fmt == "melo":
        _write_lines(base / "metadata.list",
                     [f"wavs/{u}.wav|{s}|{iso}|{t}" for u, t, s, iso in rows])

    return str(base)


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
