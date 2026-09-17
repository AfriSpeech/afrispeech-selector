"""AfriSpeech Selector — pick African languages and prepare training-ready data.

Strength is measured by recorded hours. The library lets you rank and filter
languages, select a country-balanced top-N, pull a capped sample per language
in a fixed training-ready schema, and export it.
"""

from .catalog import (
    COUNTRY_NAMES,
    DATASETS,
    DATASET_NAMES,
    DATASET_ID,
    LanguageEntry,
    by_subset,
    countries,
    load_catalog,
    resolve_dataset,
)
from .selector import filter_catalog, plan_samples, select_top
from .builder import (
    SCHEMAS,
    STANDARD_COLUMNS,
    apply_schema,
    build_dataset,
    build_subset,
    stream_dataset,
)
from .export import (
    TTS_FORMATS,
    export_archive,
    export_metadata_csv,
    export_parquet,
    export_tts,
    push_to_hub,
)

__all__ = [
    "DATASET_ID",
    "DATASETS",
    "DATASET_NAMES",
    "COUNTRY_NAMES",
    "LanguageEntry",
    "by_subset",
    "countries",
    "load_catalog",
    "resolve_dataset",
    "filter_catalog",
    "plan_samples",
    "select_top",
    "STANDARD_COLUMNS",
    "SCHEMAS",
    "apply_schema",
    "build_dataset",
    "build_subset",
    "stream_dataset",
    "TTS_FORMATS",
    "export_archive",
    "export_metadata_csv",
    "export_parquet",
    "export_tts",
    "push_to_hub",
]

__version__ = "0.1.0"
