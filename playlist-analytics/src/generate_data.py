"""Generate a realistic SYNTHETIC Top 50 playlist snapshot dataset.

Simulates ~120 days of daily Top-50 snapshots with believable rank churn,
song lifecycles (entry -> rise -> peak -> decay -> exit) and popularity drift.
"""
from __future__ import annotations

import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 20240501
N_DAYS = 120
START = date(2024, 1, 1)

ARTISTS = [
    "Nova Reyes", "The Lantern Club", "Jaxon Frey", "Milo & The Tides",
    "Priya Anand", "Cold Harbor", "Deja Monroe", "Ruben Castellanos",
    "Saint Avery", "Kira Lindqvist", "The Paper Kites Co.", "Omar Diallo",
    "Halcyon Youth", "Bea Whitlock", "Trey Nakamura", "Velvet Static",
    "Lorna Sky", "Dust & Ember",
]

TITLE_A = ["Midnight", "Paper", "Golden", "Neon", "Hollow", "Sugar", "Static",
           "Velvet", "Ivory", "Crimson", "Silver", "Ashen", "Wildflower",
           "Lonely", "Bitter", "Sunlit", "Restless", "Faded", "Electric", "Quiet"]
TITLE_B = ["Hours", "Crowns", "Lines", "Rivers", "Machines", "Weather", "Ghosts",
           "Highways", "Letters", "Vices", "Summers", "Signals", "Rooms",
           "Promises", "Engines", "Daylight", "Wires", "Bones", "Cities", "Names"]


def _make_catalog(rng: random.Random) -> list[dict]:
    titles = set()
    while len(titles) < 150:
        titles.add(f"{rng.choice(TITLE_A)} {rng.choice(TITLE_B)}")
    catalog = []
    for i, title in enumerate(sorted(titles)):
        primary = rng.choice(ARTISTS)
        # ~18% of tracks are collaborations rendered as "A feat. B"
        if rng.random() < 0.18:
            feat = rng.choice([a for a in ARTISTS if a != primary])
            artist = f"{primary} feat. {feat}"
        else:
            artist = primary
        album_type = "Single" if rng.random() < 0.45 else "Album"
        catalog.append(
            {
                "song": title,
                "artist": artist,
                "duration_ms": int(rng.gauss(205_000, 38_000)),
                "album_type": album_type,
                "total_tracks": 1 if album_type == "Single" else rng.randint(6, 20),
                "is_explicit": rng.random() < 0.38,
                "album_cover_url": f"https://cdn.example.com/covers/{i:03d}.jpg",
                # latent quality drives peak chart strength
                "strength": rng.uniform(0.25, 1.0),
            }
        )
    return catalog


def generate(path: str = "data/sample_playlist_data.csv") -> pd.DataFrame:
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)
    catalog = _make_catalog(rng)

    # Assign each song a lifecycle window inside the observation period.
    lifecycles = []
    for track in catalog:
        life = int(np_rng.integers(45, 200))
        entry = int(np_rng.integers(-150, N_DAYS - 8))
        lifecycles.append({**track, "entry": entry, "life": life})

    rows = []
    for d in range(N_DAYS):
        snapshot_date = START + timedelta(days=d)
        scores = []
        for t in lifecycles:
            age = d - t["entry"]
            if age < 0 or age > t["life"]:
                continue
            # Lifecycle curve: quick rise, plateau, slow decay (0..1)
            frac = age / max(t["life"], 1)
            curve = np.sin(np.pi * min(frac, 1.0) ** 0.75)
            score = t["strength"] * curve + np_rng.normal(0, 0.045)
            scores.append((score, t))
        if len(scores) < 50:
            continue
        scores.sort(key=lambda x: -x[0])
        for pos, (score, t) in enumerate(scores[:50], start=1):
            # Popularity tracks chart strength but with its own noise/drift.
            popularity = int(np.clip(round(96 - 0.55 * pos + 14 * score
                                           + np_rng.normal(0, 3.2)), 1, 100))
            rows.append(
                {
                    "date": snapshot_date.isoformat(),
                    "position": pos,
                    "song": t["song"],
                    "artist": t["artist"],
                    "popularity": popularity,
                    "duration_ms": t["duration_ms"],
                    "album_type": t["album_type"],
                    "total_tracks": t["total_tracks"],
                    "is_explicit": t["is_explicit"],
                    "album_cover_url": t["album_cover_url"],
                }
            )

    df = pd.DataFrame(rows)

    # Inject a small, realistic amount of dirt so validation has work to do.
    dirty = df.sample(frac=0.012, random_state=SEED).index
    df.loc[dirty[: len(dirty) // 2], "popularity"] = np.nan
    df.loc[dirty[len(dirty) // 2:], "artist"] = (
        df.loc[dirty[len(dirty) // 2:], "artist"].str.upper().radd("  ")
    )
    df = pd.concat([df, df.sample(15, random_state=7)], ignore_index=True)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    out = generate()
    print(f"Wrote {len(out)} rows, {out['song'].nunique()} songs, "
          f"{out['artist'].nunique()} artists, {out['date'].nunique()} days")
