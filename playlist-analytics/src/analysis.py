"""STEP 3 & 4 — Analysis modules and KPI computation.

Every function takes the cleaned snapshot DataFrame (optionally already
filtered by the dashboard) and returns tidy DataFrames ready for charting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import add_row_features, artist_features, daily_features, song_features

FAST_RISER_MIN_DAYS = 5


# ---------------------------------------------------------------- STEP 4 KPIs
def kpi_summary(df: pd.DataFrame) -> dict:
    """The six headline KPIs surfaced at the top of the dashboard."""
    if df.empty:
        return {k: 0 for k in [
            "avg_days_on_chart", "avg_rank", "avg_rank_volatility",
            "popularity_trend", "top_dominance", "explicit_share",
            "unique_songs", "unique_artists", "snapshot_days",
        ]} | {"top_artist": "—", "popularity_trend_delta": 0.0}

    songs = song_features(df)
    artists = artist_features(df, songs)
    daily = daily_features(df)

    # Popularity Score Trend: mean of the latest 7 daily averages, and its
    # change versus the previous 7 days (engagement direction of travel).
    recent = daily["avg_popularity"].tail(7).mean()
    prior = daily["avg_popularity"].tail(14).head(7).mean()

    return {
        "avg_days_on_chart": round(songs["days_on_chart"].mean(), 1),
        "avg_rank": round(df["position"].mean(), 1),
        "avg_rank_volatility": round(songs["rank_volatility"].mean(), 2),
        "popularity_trend": round(recent, 1),
        "popularity_trend_delta": round(recent - prior, 2) if not np.isnan(prior) else 0.0,
        "top_dominance": float(artists["dominance_index"].iloc[0]),
        "top_artist": str(artists["primary_artist"].iloc[0]),
        # Explicit Content Share = % of all chart slots held by explicit tracks.
        "explicit_share": round(df["is_explicit"].mean() * 100, 1),
        "unique_songs": int(df["song"].nunique()),
        "unique_artists": int(df["primary_artist"].nunique()),
        "snapshot_days": int(df["date"].nunique()),
    }


# ------------------------------------------- MODULE 1: Playlist ranking
def rank_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of popularity across each rank slot (1–50)."""
    return (
        df.groupby("position")
        .agg(avg_popularity=("popularity", "mean"),
             observations=("song", "size"),
             unique_songs=("song", "nunique"))
        .reset_index()
        .round(2)
    )


def entry_exit_table(df: pd.DataFrame) -> pd.DataFrame:
    """Entry date, exit date and run length for every song."""
    s = song_features(df)
    out = s[["song", "artist", "entry_date", "exit_date", "days_on_chart",
             "best_rank", "avg_rank"]].copy()
    out["run_span_days"] = (out["exit_date"] - out["entry_date"]).dt.days + 1
    # Gap between span and days_on_chart reveals re-entries after falling off.
    out["re_entered"] = out["run_span_days"] > out["days_on_chart"]
    return out.sort_values("entry_date").reset_index(drop=True)


def movers(df: pd.DataFrame, top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fast risers vs. slow decliners, ranked by positions gained per day."""
    s = song_features(df)
    s = s[s["days_on_chart"] >= FAST_RISER_MIN_DAYS]
    cols = ["song", "artist", "days_on_chart", "best_rank", "avg_rank",
            "net_rank_gain", "rank_velocity", "avg_popularity"]
    risers = s.sort_values("rank_velocity", ascending=False).head(top_n)[cols]
    decliners = s.sort_values("rank_velocity").head(top_n)[cols]
    return risers.reset_index(drop=True), decliners.reset_index(drop=True)


def rank_movement_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Daily count of songs that climbed, fell or held their position."""
    d = add_row_features(df).dropna(subset=["rank_change"])
    d["movement"] = np.select(
        [d["rank_change"] > 0, d["rank_change"] < 0],
        ["Climbed", "Fell"], default="Held",
    )
    return (
        d.groupby(["date", "movement"]).size().rename("songs").reset_index()
    )


# ------------------------------------------- MODULE 2: Song-level performance
def longest_charting(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    s = song_features(df)
    return s.nlargest(top_n, "days_on_chart")[
        ["song", "artist", "days_on_chart", "best_rank", "avg_rank",
         "avg_popularity", "rank_volatility"]
    ].reset_index(drop=True)


def highest_popularity(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    s = song_features(df)
    return s.nlargest(top_n, "avg_popularity")[
        ["song", "artist", "avg_popularity", "peak_popularity",
         "days_on_chart", "best_rank"]
    ].reset_index(drop=True)


def peak_vs_longevity(df: pd.DataFrame) -> pd.DataFrame:
    """Compare peak-rank achievement against chart longevity.

    Quadrants (split on the median of each axis):
      Hit & Stay   - strong peak, long run   -> flagship catalogue assets
      Flash Hit    - strong peak, short run  -> spike-driven, needs re-push
      Slow Burner  - modest peak, long run   -> steady playlist earners
      Marginal     - modest peak, short run  -> low ROI
    """
    s = song_features(df).copy()
    if s.empty:
        s["quadrant"] = []
        return s
    med_rank = s["best_rank"].median()
    med_days = s["days_on_chart"].median()
    strong = s["best_rank"] <= med_rank
    long_run = s["days_on_chart"] >= med_days
    s["quadrant"] = np.select(
        [strong & long_run, strong & ~long_run, ~strong & long_run],
        ["Hit & Stay", "Flash Hit", "Slow Burner"], default="Marginal",
    )
    return s


# ------------------------------------------- MODULE 3: Artist performance
def artist_leaderboard(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    return artist_features(df).head(top_n)


def artist_dominance_over_time(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Weekly share of Top-50 slots held by each of the leading artists."""
    lead = artist_features(df).head(top_n)["primary_artist"].tolist()
    d = df.copy()
    d["week"] = d["date"].dt.to_period("W").dt.start_time
    weekly = d.groupby(["week", "primary_artist"]).size().rename("slots").reset_index()
    total = weekly.groupby("week")["slots"].transform("sum")
    weekly["share_pct"] = (weekly["slots"] / total * 100).round(2)
    return weekly[weekly["primary_artist"].isin(lead)].reset_index(drop=True)


# ------------------------------------------- MODULE 4: Popularity analytics
def popularity_rank_correlation(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": len(df)}
    return {
        "pearson": round(df["popularity"].corr(df["position"]), 3),
        "spearman": round(df["popularity"].corr(df["position"], method="spearman"), 3),
        "n": int(len(df)),
    }


def popularity_by_band(df: pd.DataFrame) -> pd.DataFrame:
    d = add_row_features(df)
    out = d.groupby("rank_band", observed=True)["popularity"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).reset_index().round(2)
    return out.rename(columns={"rank_band": "Rank band"})


def stability_vs_volatility(df: pd.DataFrame) -> pd.DataFrame:
    s = song_features(df)
    return s[["song", "artist", "rank_volatility", "popularity_volatility",
              "avg_popularity", "days_on_chart", "best_rank"]]


# ------------------------------------------- MODULE 5: Content attributes
def explicit_comparison(df: pd.DataFrame) -> pd.DataFrame:
    s = song_features(df)
    out = s.groupby("is_explicit").agg(
        songs=("song", "count"),
        avg_rank=("avg_rank", "mean"),
        avg_best_rank=("best_rank", "mean"),
        avg_days_on_chart=("days_on_chart", "mean"),
        avg_popularity=("avg_popularity", "mean"),
        avg_volatility=("rank_volatility", "mean"),
    ).reset_index().round(2)
    out["is_explicit"] = out["is_explicit"].map({True: "Explicit", False: "Clean"})
    return out.rename(columns={"is_explicit": "Content"})


def album_type_comparison(df: pd.DataFrame) -> pd.DataFrame:
    s = song_features(df)
    return s.groupby("album_type").agg(
        songs=("song", "count"),
        avg_rank=("avg_rank", "mean"),
        avg_best_rank=("best_rank", "mean"),
        avg_days_on_chart=("days_on_chart", "mean"),
        avg_popularity=("avg_popularity", "mean"),
    ).reset_index().round(2)


def duration_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Song-level duration table plus performance grouped into duration bins."""
    s = song_features(df).copy()
    s["duration_bucket"] = pd.cut(
        s["duration_min"],
        bins=[0, 2.5, 3.0, 3.5, 4.0, 100],
        labels=["<2:30", "2:30–3:00", "3:00–3:30", "3:30–4:00", "4:00+"],
    )
    grouped = s.groupby("duration_bucket", observed=True).agg(
        songs=("song", "count"),
        avg_rank=("avg_rank", "mean"),
        avg_popularity=("avg_popularity", "mean"),
        avg_days_on_chart=("days_on_chart", "mean"),
    ).reset_index().round(2)
    return s, grouped


def album_size_analysis(df: pd.DataFrame) -> pd.DataFrame:
    s = song_features(df)
    s = s[s["album_type"].str.lower() != "single"].copy()
    if s.empty:
        return pd.DataFrame(columns=["tracks_bucket", "songs", "avg_rank",
                                     "avg_popularity", "avg_days_on_chart"])
    s["tracks_bucket"] = pd.cut(
        s["total_tracks"], bins=[0, 8, 12, 16, 100],
        labels=["1–8", "9–12", "13–16", "17+"],
    )
    return s.groupby("tracks_bucket", observed=True).agg(
        songs=("song", "count"),
        avg_rank=("avg_rank", "mean"),
        avg_popularity=("avg_popularity", "mean"),
        avg_days_on_chart=("days_on_chart", "mean"),
    ).reset_index().round(2)
