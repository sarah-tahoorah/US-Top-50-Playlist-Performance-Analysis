"""STEP 2 — Feature engineering (song level, artist level, daily level)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLLING_WINDOW = 7  # days — popularity trend smoothing window


def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level derived columns added to the daily snapshot table."""
    out = df.copy()
    out["duration_min"] = out["duration_ms"] / 60_000.0
    # Rank band used for Top-10/20/50 comparisons.
    out["rank_band"] = pd.cut(
        out["position"], bins=[0, 10, 20, 50],
        labels=["Top 10", "Top 11-20", "Top 21-50"],
    )
    out = out.sort_values(["song", "artist", "date"])
    grp = out.groupby(["song", "artist"], sort=False)
    # 7-day rolling mean popularity = Popularity Trend Score (per song/day).
    out["popularity_trend"] = (
        grp["popularity"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean())
        .round(2)
    )
    # Positive rank_change = the song moved UP the chart since the prior day.
    out["rank_change"] = grp["position"].transform(lambda s: s.shift(1) - s)
    return out.sort_values(["date", "position"]).reset_index(drop=True)


def song_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (song, artist) with the Step-2 longevity/performance metrics."""
    d = add_row_features(df)
    g = d.groupby(["song", "artist"], sort=False)

    feats = g.agg(
        primary_artist=("primary_artist", "first"),
        days_on_chart=("date", "nunique"),          # longevity
        avg_rank=("position", "mean"),
        best_rank=("position", "min"),              # peak achievement
        worst_rank=("position", "max"),
        rank_volatility=("position", "std"),        # Rank Volatility Index (std dev)
        avg_popularity=("popularity", "mean"),
        peak_popularity=("popularity", "max"),
        popularity_volatility=("popularity", "std"),
        entry_date=("date", "min"),
        exit_date=("date", "max"),
        duration_min=("duration_min", "first"),
        album_type=("album_type", "first"),
        total_tracks=("total_tracks", "first"),
        is_explicit=("is_explicit", "first"),
        album_cover_url=("album_cover_url", "first"),
    ).reset_index()

    # Final Popularity Trend Score = last observed 7-day rolling mean.
    last_trend = (
        d.sort_values("date").groupby(["song", "artist"])["popularity_trend"].last()
        .rename("popularity_trend_score").reset_index()
    )
    feats = feats.merge(last_trend, on=["song", "artist"], how="left")

    # Net momentum: rank improvement from first to last appearance
    # (positive = climbed). Divided by days to express positions gained/day.
    first_last = d.sort_values("date").groupby(["song", "artist"])["position"].agg(
        ["first", "last"]
    )
    feats = feats.merge(
        (first_last["first"] - first_last["last"]).rename("net_rank_gain"),
        on=["song", "artist"], how="left",
    )
    feats["rank_velocity"] = (
        feats["net_rank_gain"] / feats["days_on_chart"].clip(lower=1)
    ).round(3)

    feats["rank_volatility"] = feats["rank_volatility"].fillna(0.0)
    feats["popularity_volatility"] = feats["popularity_volatility"].fillna(0.0)
    for col in ["avg_rank", "avg_popularity", "peak_popularity",
                "rank_volatility", "popularity_volatility", "duration_min"]:
        feats[col] = feats[col].round(2)

    return feats.sort_values("avg_rank").reset_index(drop=True)


def artist_features(df: pd.DataFrame, songs: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per primary artist, including the Artist Dominance Index."""
    songs = song_features(df) if songs is None else songs
    d = df.copy()

    per_artist = d.groupby("primary_artist").agg(
        chart_days=("date", "count"),               # total song-days on chart
        unique_songs=("song", "nunique"),
        avg_rank=("position", "mean"),
        best_rank=("position", "min"),
        avg_popularity=("popularity", "mean"),
        top10_days=("position", lambda s: int((s <= 10).sum())),
        first_seen=("date", "min"),
        last_seen=("date", "max"),
    ).reset_index()

    per_artist["explicit_share"] = (
        d.groupby("primary_artist")["is_explicit"].mean().values * 100
    ).round(1)

    # --- Artist Dominance Index (ADI) -------------------------------------
    # Weighted blend of three min-max normalised components, scaled 0-100:
    #   50%  chart presence   : total song-days on the chart
    #   30%  catalogue breadth: unique songs charted
    #   20%  rank quality     : inverted average rank (rank 1 -> 1.0, rank 50 -> 0)
    def _norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng else pd.Series(1.0, index=s.index)

    rank_quality = (50 - per_artist["avg_rank"]) / 49.0
    per_artist["dominance_index"] = (
        100 * (
            0.50 * _norm(per_artist["chart_days"])
            + 0.30 * _norm(per_artist["unique_songs"])
            + 0.20 * rank_quality.clip(0, 1)
        )
    ).round(1)

    per_artist["avg_rank"] = per_artist["avg_rank"].round(2)
    per_artist["avg_popularity"] = per_artist["avg_popularity"].round(2)
    return per_artist.sort_values("dominance_index", ascending=False).reset_index(drop=True)


def daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per snapshot date — chart-level aggregates."""
    d = df.copy()
    daily = d.groupby("date").agg(
        songs=("song", "nunique"),
        artists=("primary_artist", "nunique"),
        avg_popularity=("popularity", "mean"),
        median_popularity=("popularity", "median"),
        explicit_share=("is_explicit", "mean"),
        avg_duration_min=("duration_ms", lambda s: s.mean() / 60_000),
    ).reset_index()
    daily["explicit_share"] = (daily["explicit_share"] * 100).round(2)
    daily["avg_popularity"] = daily["avg_popularity"].round(2)
    daily["avg_duration_min"] = daily["avg_duration_min"].round(2)

    # Chart churn: songs present today that were absent yesterday.
    sets = d.groupby("date")["song"].apply(set)
    churn = [np.nan]
    dates = list(sets.index)
    for i in range(1, len(dates)):
        churn.append(len(sets.iloc[i] - sets.iloc[i - 1]))
    daily["new_entries"] = churn
    return daily
