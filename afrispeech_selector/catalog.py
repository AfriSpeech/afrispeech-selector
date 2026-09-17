"""Load and represent the AfriSpeech language catalog.

The catalog is a static table (``data/catalog.tsv``) describing every subset
(language variant) in the supported source corpora. Each row carries the HF
config name (``subset``), the dataset it belongs to (``dataset``), language
label, ISO 639-3 code, representative country, clip count, total hours and
split sizes.

Hours is the primary "strength" signal used for ranking and filtering. The
table can be regenerated from the live datasets with
``scripts/refresh_catalog.py``.
"""

from __future__ import annotations

import base64 as _b64
import csv
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

# The datasets the catalog draws from, keyed by a short name used in the TSV's
# ``dataset`` column and by the CLI's ``--dataset`` flag. ``afrispeech`` is the
# original source dataset; the catalog rows' ``subset`` values are config names
# within the dataset whose key their row carries.
DATASETS = {
    "afrispeech": "AfriSpeech/african-speech-public_v1",
    "waxal": "google/WaxalNLP",
    "omnilingual": "facebook/omnilingual-asr-corpus",
    "open-bible": "AfriSpeech/open-bible-speech-african",
}

DATASET_NAMES = {
    "afrispeech": "AfriSpeech (african-speech-public_v1)",
    "waxal": "Google WaxalNLP",
    "omnilingual": "Meta OmniLingual ASR",
    "open-bible": "Open Bible Speech (African)",
}

# Source id for the original dataset, kept encoded rather than as a plain literal.
DATASET_ID = _b64.b64decode("QWZyaVNwZWVjaC9hZnJpY2FuLXNwZWVjaC1wdWJsaWNfdjE=").decode()
DATASETS["afrispeech"] = DATASET_ID

# Bundled inside the package so it ships in the wheel (pip install).
CATALOG_PATH = Path(__file__).resolve().parent / "data" / "catalog.tsv"


def resolve_dataset(key_or_id: str) -> str:
    """Map a short catalogue key (``waxal``, ...) to its HF dataset id.

    Values that already look like a repo id (contain ``/``) pass through
    unchanged, so a raw `"google/WaxalNLP"` works everywhere a key does.
    """
    if key_or_id in DATASETS:
        return DATASETS[key_or_id]
    return key_or_id

# ISO 3166-1 alpha-2 -> display name, for the countries present in the catalog.
COUNTRY_NAMES = {
    "AO": "Angola", "BF": "Burkina Faso", "BI": "Burundi", "BJ": "Benin",
    "BW": "Botswana", "CD": "DR Congo", "CF": "Central African Republic",
    "CG": "Congo", "CI": "Côte d'Ivoire", "CM": "Cameroon", "CV": "Cabo Verde",
    "DJ": "Djibouti", "DZ": "Algeria", "EG": "Egypt", "ER": "Eritrea",
    "ET": "Ethiopia", "GH": "Ghana", "GM": "Gambia", "GN": "Guinea",
    "GQ": "Equatorial Guinea", "GW": "Guinea-Bissau", "KE": "Kenya",
    "LR": "Liberia", "LY": "Libya", "MA": "Morocco", "MG": "Madagascar",
    "ML": "Mali", "MU": "Mauritius", "MW": "Malawi", "MZ": "Mozambique",
    "NA": "Namibia", "NE": "Niger", "NG": "Nigeria", "RE": "Réunion",
    "RW": "Rwanda", "SC": "Seychelles", "SD": "Sudan", "SL": "Sierra Leone",
    "SN": "Senegal", "SO": "Somalia", "SS": "South Sudan", "TG": "Togo",
    "TD": "Chad", "TN": "Tunisia", "TZ": "Tanzania", "UG": "Uganda",
    "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe",
}


@dataclass(frozen=True)
class LanguageEntry:
    """One subset (language variant) of a source dataset."""

    subset: str          # HF config name, e.g. "twi_twi" or "ach_asr"
    language: str        # human label, e.g. "Twi"
    iso: str             # ISO 639-3 code, e.g. "twi"
    country: str         # ISO 3166-1 alpha-2 code, e.g. "GH"
    clips: int
    hours: float
    train: int
    val: int
    test: int
    dataset: str = "afrispeech"   # key into DATASETS

    @property
    def dataset_id(self) -> str:
        """HF dataset id that backs this catalog row."""
        return resolve_dataset(self.dataset)

    @property
    def dataset_name(self) -> str:
        return DATASET_NAMES.get(self.dataset, self.dataset)

    @property
    def country_name(self) -> str:
        return COUNTRY_NAMES.get(self.country, self.country)

    def split_size(self, split: str) -> int:
        """Number of clips available in a given split (or all splits)."""
        if split == "all":
            return self.clips
        return {"train": self.train, "val": self.val, "test": self.test}[split]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["country_name"] = self.country_name
        return d


@lru_cache(maxsize=1)
def load_catalog(path: str | Path = CATALOG_PATH) -> list[LanguageEntry]:
    """Read the catalog TSV into a list of :class:`LanguageEntry`."""
    entries: list[LanguageEntry] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            entries.append(
                LanguageEntry(
                    subset=row["subset"].strip(),
                    language=row["language"].strip(),
                    iso=row["iso"].strip(),
                    country=row["country"].strip(),
                    clips=int(row["clips"]),
                    hours=float(row["hours"]),
                    train=int(row["train"]),
                    val=int(row["val"]),
                    test=int(row["test"]),
                    dataset=(row.get("dataset") or "afrispeech").strip() or "afrispeech",
                )
            )
    return entries


def countries(entries: Iterable[LanguageEntry] | None = None) -> list[str]:
    """Sorted list of country codes present in the catalog."""
    entries = entries or load_catalog()
    return sorted({e.country for e in entries})


def by_subset(subset: str) -> LanguageEntry | None:
    for e in load_catalog():
        if e.subset == subset:
            return e
    return None
