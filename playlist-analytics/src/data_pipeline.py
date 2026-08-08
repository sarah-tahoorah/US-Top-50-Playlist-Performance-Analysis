"""STEP 1 — Data ingestion, validation and cleaning.

Public API:
    load_raw(path)   -> pd.DataFrame
    clean(df)        -> (clean_df, QualityReport)
    load_clean(path) -> (clean_df, QualityReport)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict

import pandas as pd

DEFAULT_DATA_PATH = os.path.join("data", "sample_playlist_data.csv")

REQUIRED_COLUMNS = [
    "date", "position", "song", "artist", "popularity", "duration_ms",
    "album_type", "total_tracks", "is_explicit", "album_cover_url",
]


@dataclass
class QualityReport:
    """Counts of data-quality issues found (and how they were resolved)."""

    rows_in: int = 0
    rows_out: int = 0
    missing_columns: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)
    invalid_positions: int = 0
    duplicate_keys: int = 0
    artists_standardised: int = 0
    unparseable_dates: int = 0
    popularity_out_of_range: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def as_frame(self) -> pd.DataFrame:
        rows = [
            ("Rows read", self.rows_in, "—"),
            ("Rows after cleaning", self.rows_out, "—"),
            ("Missing required columns", len(self.missing_columns),
             ", ".join(self.missing_columns) or "none"),
            ("Cells with missing values", sum(self.missing_values.values()),
             "popularity imputed per-song; other gaps retained"),
            ("Positions outside 1–50", self.invalid_positions, "dropped"),
            ("Popularity outside 0–100", self.popularity_out_of_range, "clipped"),
            ("Unparseable dates", self.unparseable_dates, "dropped"),
            ("Duplicate (song, artist, date)", self.duplicate_keys,
             "kept best (lowest) position"),
            ("Artist names standardised", self.artists_standardised, "normalised"),
        ]
        return pd.DataFrame(rows, columns=["Check", "Count", "Resolution"])


# --- artist name standardisation -------------------------------------------

_FEAT_PATTERN = re.compile(
    r"\s*(?:feat\.?|featuring|ft\.?|with)\s+", flags=re.IGNORECASE
)


def standardise_artist(raw: str) -> str:
    """Trim whitespace, collapse inner spaces, normalise 'feat.' separators and
    apply consistent Title Case while preserving intentional short words."""
    if not isinstance(raw, str):
        return ""
    name = re.sub(r"\s+", " ", raw).strip()
    # Unify every featuring variant to a single canonical " feat. " separator.
    name = _FEAT_PATTERN.sub(" feat. ", name)
    parts = [p.strip() for p in name.split(" feat. ") if p.strip()]
    fixed = []
    for part in parts:
        # Only re-case ALL CAPS / all lower input; leave mixed case authored
        # names (e.g. "The Lantern Club") untouched.
        if part.isupper() or part.islower():
            part = " ".join(
                w if w.lower() in {"&", "and", "the", "of"} and i > 0 else w.capitalize()
                for i, w in enumerate(part.split(" "))
            )
        fixed.append(part)
    return " feat. ".join(fixed)


def primary_artist(name: str) -> str:
    """The lead (billed-first) artist, used for artist-level aggregation."""
    return name.split(" feat. ")[0].strip()


# --- pipeline ---------------------------------------------------------------

def load_raw(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path!r}. Run "
            "`python -m src.generate_data` to create the synthetic sample."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    rep = QualityReport(rows_in=len(df))
    rep.missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if rep.missing_columns:
        raise ValueError(f"Dataset missing required columns: {rep.missing_columns}")

    df = df.copy()
    rep.missing_values = {
        c: int(n) for c, n in df[REQUIRED_COLUMNS].isna().sum().items() if n
    }

    # Dates -------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    rep.unparseable_dates = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])

    # Position must be an integer within the Top-50 window ---------------
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    bad_pos = df["position"].isna() | (df["position"] < 1) | (df["position"] > 50)
    rep.invalid_positions = int(bad_pos.sum())
    df = df.loc[~bad_pos].copy()
    df["position"] = df["position"].astype(int)

    # Text fields ---------------------------------------------------------
    df["song"] = df["song"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    original_artist = df["artist"].astype(str)
    df["artist"] = original_artist.map(standardise_artist)
    rep.artists_standardised = int((df["artist"] != original_artist).sum())
    df["primary_artist"] = df["artist"].map(primary_artist)

    # Popularity: impute per-song median, then clip to the valid 0–100 range.
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["popularity"] = df["popularity"].fillna(
        df.groupby(["song", "artist"])["popularity"].transform("median")
    )
    df["popularity"] = df["popularity"].fillna(df["popularity"].median())
    rep.popularity_out_of_range = int(
        ((df["popularity"] < 0) | (df["popularity"] > 100)).sum()
    )
    df["popularity"] = df["popularity"].clip(0, 100)

    # Attributes -----------------------------------------------------------
    df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce")
    df["total_tracks"] = pd.to_numeric(df["total_tracks"], errors="coerce")
    df["is_explicit"] = (
        df["is_explicit"].astype(str).str.strip().str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )
    df["album_type"] = (
        df["album_type"].astype(str).str.strip().str.capitalize()
        .replace({"Nan": "Unknown"})
    )

    # Duplicates: one (song, artist, date) row wins — the best rank observed.
    key = ["song", "artist", "date"]
    rep.duplicate_keys = int(df.duplicated(subset=key).sum())
    df = (
        df.sort_values("position")
        .drop_duplicates(subset=key, keep="first")
        .sort_values(["date", "position"])
        .reset_index(drop=True)
    )

    rep.rows_out = len(df)
    return df, rep


def load_clean(path: str = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, QualityReport]:
    return clean(load_raw(path))


if __name__ == "__main__":
    frame, report = load_clean()
    print(report.as_frame().to_string(index=False))
