"""United States Top 50 Playlist Performance & Song Popularity Trend Analysis.

Open-access Streamlit dashboard.
Run with:  streamlit run app.py
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analysis as an
from src import features as fe
from src.data_pipeline import DEFAULT_DATA_PATH, load_clean

st.set_page_config(
    page_title="US Top 50 Playlist Analytics",
    page_icon="🎧",
    layout="wide",
)

PLOT_TEMPLATE = "plotly_white"


# --------------------------------------------------------------- data loading
@st.cache_data(show_spinner="Loading playlist snapshots…")
def get_data(path: str = DEFAULT_DATA_PATH):
    df, report = load_clean(path)
    return df, report.as_frame(), report.as_dict()


def empty_state(message: str = "No rows match the current filters.") -> None:
    st.warning(f"{message} Try widening the date range, rank range or filters.")


# ------------------------------------------------------------------- sidebar
if not os.path.exists(DEFAULT_DATA_PATH):
    st.error(
        f"Dataset not found at `{DEFAULT_DATA_PATH}`. "
        "Run `python -m src.generate_data` to create the synthetic sample."
    )
    st.stop()

raw, quality_frame, quality = get_data()

st.sidebar.header("Filters")
min_date = raw["date"].min().date()
max_date = raw["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

rank_range = st.sidebar.slider("Rank range", 1, 50, (1, 50))

artist_options = sorted(raw["primary_artist"].unique())
sel_artists = st.sidebar.multiselect("Artists", artist_options, default=[])

song_pool = raw[raw["primary_artist"].isin(sel_artists)] if sel_artists else raw
sel_songs = st.sidebar.multiselect("Songs", sorted(song_pool["song"].unique()), default=[])

album_types = sorted(raw["album_type"].unique())
sel_album = st.sidebar.multiselect("Album type", album_types, default=album_types)

explicit_choice = st.sidebar.radio(
    "Content", ["All", "Explicit only", "Clean only"], horizontal=False
)

mask = (
    (raw["date"].dt.date >= start_date)
    & (raw["date"].dt.date <= end_date)
    & (raw["position"].between(*rank_range))
    & (raw["album_type"].isin(sel_album))
)
if sel_artists:
    mask &= raw["primary_artist"].isin(sel_artists)
if sel_songs:
    mask &= raw["song"].isin(sel_songs)
if explicit_choice == "Explicit only":
    mask &= raw["is_explicit"]
elif explicit_choice == "Clean only":
    mask &= ~raw["is_explicit"]

df = raw.loc[mask].copy()

st.sidebar.caption(f"{len(df):,} of {len(raw):,} snapshot rows selected")

# --------------------------------------------------------------------- header
st.title("🎧 United States Top 50 — Playlist Performance & Popularity Trends")
st.caption(
    "Historical analytics of daily Top-50 playlist snapshots for artist promotion, "
    "release timing and marketing-spend decisions. Data is synthetic sample data."
)

if df.empty:
    empty_state()
    st.stop()

# ----------------------------------------------------------------- STEP 4 KPIs
k = an.kpi_summary(df)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Avg days on chart", k["avg_days_on_chart"], help="Longevity indicator")
c2.metric("Average rank", k["avg_rank"], help="Overall chart performance (lower is better)")
c3.metric("Rank volatility index", k["avg_rank_volatility"],
          help="Mean std-dev of daily rank per song — lower is more stable")
c4.metric("Popularity trend (7d)", k["popularity_trend"],
          delta=k["popularity_trend_delta"], help="7-day mean popularity vs prior 7 days")
c5.metric("Top dominance index", k["top_dominance"], help=f"Leader: {k['top_artist']}")
c6.metric("Explicit share", f"{k['explicit_share']}%",
          help="% of chart slots held by explicit tracks")

st.caption(
    f"{k['unique_songs']} songs · {k['unique_artists']} artists · "
    f"{k['snapshot_days']} snapshot days in the current selection"
)

tabs = st.tabs([
    "Playlist timeline",
    "Song performance",
    "Artist dominance",
    "Popularity analytics",
    "Content attributes",
    "Data quality",
])

# ============================================== 1. Playlist timeline explorer
with tabs[0]:
    st.subheader("Playlist timeline explorer")
    daily = fe.daily_features(df)

    fig = px.line(
        daily, x="date", y="avg_popularity",
        title="Average popularity of the charting set, by day",
        labels={"date": "Snapshot date", "avg_popularity": "Average popularity"},
        template=PLOT_TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            daily, x="date", y="new_entries",
            title="Daily chart churn — new entries vs. previous day",
            labels={"date": "Snapshot date", "new_entries": "New songs"},
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        move = an.rank_movement_profile(df)
        if move.empty:
            empty_state("Not enough consecutive days to compute rank movement.")
        else:
            fig = px.area(
                move, x="date", y="songs", color="movement",
                title="Rank movement pattern — climbed / held / fell",
                labels={"date": "Snapshot date", "songs": "Songs", "movement": "Movement"},
                template=PLOT_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Rank slot profile (1–50)")
    dist = an.rank_distribution(df)
    fig = px.bar(
        dist, x="position", y="avg_popularity",
        title="Average popularity by chart position",
        labels={"position": "Chart position", "avg_popularity": "Average popularity"},
        template=PLOT_TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Fast risers vs. slow decliners")
    risers, decliners = an.movers(df)
    a, b = st.columns(2)
    a.caption("Fast risers (positions gained per charting day)")
    a.dataframe(risers, use_container_width=True, hide_index=True)
    b.caption("Slow decliners")
    b.dataframe(decliners, use_container_width=True, hide_index=True)

    with st.expander("Entry / exit dates for every song"):
        st.dataframe(an.entry_exit_table(df), use_container_width=True, hide_index=True)

# ================================================ 2. Song ranking trend charts
with tabs[1]:
    st.subheader("Song-level performance")
    longest = an.longest_charting(df)
    a, b = st.columns(2)
    with a:
        fig = px.bar(
            longest.sort_values("days_on_chart"), x="days_on_chart", y="song",
            orientation="h", title="Longest-charting songs",
            labels={"days_on_chart": "Days on chart", "song": "Song"},
            template=PLOT_TEMPLATE, height=520,
        )
        st.plotly_chart(fig, use_container_width=True)
    with b:
        pop = an.highest_popularity(df)
        fig = px.bar(
            pop.sort_values("avg_popularity"), x="avg_popularity", y="song",
            orientation="h", title="Highest average popularity",
            labels={"avg_popularity": "Average popularity", "song": "Song"},
            template=PLOT_TEMPLATE, height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Peak rank vs. longevity")
    quad = an.peak_vs_longevity(df)
    fig = px.scatter(
        quad, x="days_on_chart", y="best_rank", color="quadrant",
        size="avg_popularity", hover_name="song", hover_data=["artist", "avg_rank"],
        title="Peak rank achieved vs. days on chart (bubble = average popularity)",
        labels={"days_on_chart": "Days on chart", "best_rank": "Best rank achieved",
                "quadrant": "Segment"},
        template=PLOT_TEMPLATE,
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Rank trajectory explorer")
    default_songs = longest["song"].head(3).tolist()
    picks = st.multiselect("Songs to plot", sorted(df["song"].unique()), default=default_songs)
    if picks:
        traj = df[df["song"].isin(picks)]
        fig = px.line(
            traj, x="date", y="position", color="song", markers=False,
            title="Daily chart position over time",
            labels={"date": "Snapshot date", "position": "Chart position", "song": "Song"},
            template=PLOT_TEMPLATE,
        )
        fig.update_yaxes(autorange="reversed", range=[50, 1])
        st.plotly_chart(fig, use_container_width=True)

        trend = fe.add_row_features(traj)
        fig = px.line(
            trend, x="date", y="popularity_trend", color="song",
            title="7-day rolling popularity trend score",
            labels={"date": "Snapshot date", "popularity_trend": "Popularity (7d rolling)",
                    "song": "Song"},
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        empty_state("No songs selected for the trajectory chart.")

    with st.expander("Full song feature table"):
        st.dataframe(fe.song_features(df), use_container_width=True, hide_index=True)

# ================================================= 3. Artist dominance leaderboard
with tabs[2]:
    st.subheader("Artist dominance leaderboard")
    st.caption(
        "Artist Dominance Index = 100 × (0.50 × normalised chart-days + "
        "0.30 × normalised unique songs + 0.20 × inverted average rank)."
    )
    board = an.artist_leaderboard(df)
    fig = px.bar(
        board.sort_values("dominance_index"), x="dominance_index", y="primary_artist",
        orientation="h", color="unique_songs",
        title="Artist Dominance Index",
        labels={"dominance_index": "Dominance index (0–100)",
                "primary_artist": "Artist", "unique_songs": "Unique songs"},
        template=PLOT_TEMPLATE, height=600,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(board, use_container_width=True, hide_index=True)

    st.markdown("#### Dominance over time")
    over_time = an.artist_dominance_over_time(df)
    if over_time.empty:
        empty_state()
    else:
        fig = px.area(
            over_time, x="week", y="share_pct", color="primary_artist",
            title="Weekly share of Top-50 slots held (leading artists)",
            labels={"week": "Week", "share_pct": "Share of chart slots (%)",
                    "primary_artist": "Artist"},
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================================================== 4. Popularity vs rank
with tabs[3]:
    st.subheader("Popularity score analytics")
    corr = an.popularity_rank_correlation(df)
    a, b, c = st.columns(3)
    a.metric("Pearson r (popularity vs rank)", corr["pearson"])
    b.metric("Spearman ρ", corr["spearman"])
    c.metric("Observations", f"{corr['n']:,}")
    st.caption(
        "A negative coefficient is expected: better (numerically lower) chart "
        "positions coincide with higher popularity scores."
    )

    fig = px.scatter(
        df.sample(min(len(df), 4000), random_state=1),
        x="position", y="popularity", color="album_type", opacity=0.45,
        trendline="ols", trendline_scope="overall",
        title="Popularity vs. chart position",
        labels={"position": "Chart position", "popularity": "Popularity score",
                "album_type": "Album type"},
        template=PLOT_TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)

    bands = an.popularity_by_band(df)
    a, b = st.columns([2, 3])
    a.markdown("#### Popularity by rank band")
    a.dataframe(bands, use_container_width=True, hide_index=True)
    banded = fe.add_row_features(df)
    fig = px.box(
        banded, x="rank_band", y="popularity", color="rank_band",
        title="Popularity distribution across Top 10 / 11–20 / 21–50",
        labels={"rank_band": "Rank band", "popularity": "Popularity score"},
        template=PLOT_TEMPLATE,
    )
    b.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Popularity stability vs. rank volatility")
    stab = an.stability_vs_volatility(df)
    fig = px.scatter(
        stab, x="rank_volatility", y="popularity_volatility",
        size="days_on_chart", color="avg_popularity", hover_name="song",
        hover_data=["artist", "best_rank"],
        title="Rank volatility vs. popularity volatility (bubble = days on chart)",
        labels={"rank_volatility": "Rank volatility index (std dev of rank)",
                "popularity_volatility": "Popularity std dev",
                "avg_popularity": "Avg popularity"},
        template=PLOT_TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================== 5. Content attributes
with tabs[4]:
    st.subheader("Content attribute analysis")

    exp = an.explicit_comparison(df)
    alb = an.album_type_comparison(df)
    a, b = st.columns(2)
    a.markdown("#### Explicit vs. clean")
    a.dataframe(exp, use_container_width=True, hide_index=True)
    b.markdown("#### Single vs. album")
    b.dataframe(alb, use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    with a:
        fig = px.bar(
            exp, x="Content", y="avg_days_on_chart", color="Content",
            title="Average days on chart — explicit vs. clean",
            labels={"avg_days_on_chart": "Average days on chart"},
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.bar(
            alb, x="album_type", y="avg_popularity", color="album_type",
            title="Average popularity — single vs. album",
            labels={"album_type": "Album type", "avg_popularity": "Average popularity"},
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Explicit share of the chart over time")
    daily = fe.daily_features(df)
    fig = px.line(
        daily, x="date", y="explicit_share",
        title="Explicit content share of Top-50 slots",
        labels={"date": "Snapshot date", "explicit_share": "Explicit share (%)"},
        template=PLOT_TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Duration vs. performance")
    songs_dur, dur_bins = an.duration_analysis(df)
    a, b = st.columns(2)
    with a:
        fig = px.scatter(
            songs_dur, x="duration_min", y="avg_popularity", color="album_type",
            size="days_on_chart", hover_name="song",
            title="Track duration vs. average popularity",
            labels={"duration_min": "Duration (minutes)",
                    "avg_popularity": "Average popularity", "album_type": "Album type"},
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.bar(
            dur_bins, x="duration_bucket", y="avg_rank", color="duration_bucket",
            title="Average rank by duration band (lower is better)",
            labels={"duration_bucket": "Duration band", "avg_rank": "Average rank"},
            template=PLOT_TEMPLATE,
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(dur_bins, use_container_width=True, hide_index=True)

    st.markdown("#### Album size (total tracks) vs. success")
    size_tbl = an.album_size_analysis(df)
    if size_tbl.empty:
        empty_state("No album tracks in the current selection.")
    else:
        fig = px.bar(
            size_tbl, x="tracks_bucket", y="avg_days_on_chart", color="tracks_bucket",
            title="Average days on chart by album size",
            labels={"tracks_bucket": "Tracks on parent album",
                    "avg_days_on_chart": "Average days on chart"},
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(size_tbl, use_container_width=True, hide_index=True)

# ======================================================= 6. Data quality report
with tabs[5]:
    st.subheader("Data quality report (STEP 1)")
    st.dataframe(quality_frame, use_container_width=True, hide_index=True)
    st.caption(
        "Validation covers the 1–50 position window, missing values, duplicate "
        "(song, artist, date) keys and artist-name standardisation."
    )
    st.download_button(
        "Download filtered snapshot data (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_playlist_data.csv",
        mime="text/csv",
    )
