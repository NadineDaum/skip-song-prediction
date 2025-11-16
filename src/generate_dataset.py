"""
Creates a synthetic skip-prediction dataset inspired by real findings from
Spotify's user-behavior research. The goal is to simulate realistic patterns
(user habits, track attributes, temporal effects) while keeping full control
and transparency over the data-generation process.

This script is part of the Skip Song Prediction portfolio project (SQL + ML).
Author: Nadine Daum
"""

# Imports 
import os
import numpy as np
import pandas as pd

import streamlit_app

# For reproducibility
RNG = np.random.default_rng(seed=42)

# Output directory 
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "synthetic_sessions.csv")

# Create data/ folder if it does not exist
os.makedirs(DATA_DIR, exist_ok=True)

print(f"[INFO] Output will be saved to: {OUTPUT_FILE}")

# Define dataset dimensions 
N_USERS = 2000
N_TRACKS = 5000
N_SESSIONS = 20000
AVG_EVENTS_PER_SESSION = 15


# Helper distributions: Age groups with approximate global streaming proportions
AGE_GROUPS = ["13-17", "18-24", "25-34", "35-44", "45+"]
AGE_WEIGHTS = [0.10, 0.35, 0.30, 0.15, 0.10]

GENDERS = ["female", "male", "non_binary"]
GENDER_WEIGHTS = [0.48, 0.48, 0.04]  # approximate modern dataset split

COUNTRIES = ["NA", "EU_UK", "LATAM", "ASIA", "AFRICA"]
COUNTRY_WEIGHTS = [0.20, 0.35, 0.20, 0.20, 0.05]

SUB_TYPES = ["free", "premium", "family", "student"]
SUB_WEIGHTS = [0.45, 0.35, 0.10, 0.10]

PLATFORMS = ["ios", "android", "desktop", "web"]
PLATFORM_WEIGHTS = [0.40, 0.40, 0.15, 0.05]

SKIP_TENDENCY = ["low", "medium", "high"]
SKIP_WEIGHTS = [0.30, 0.50, 0.20]  # most users are moderate skippers


# Create Users table 
def generate_users(n_users: int):
    """Create synthetic user-level attributes."""
    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "age_group": RNG.choice(AGE_GROUPS, size=n_users, p=AGE_WEIGHTS),
        "gender": RNG.choice(GENDERS, size=n_users, p=GENDER_WEIGHTS),
        "country": RNG.choice(COUNTRIES, size=n_users, p=COUNTRY_WEIGHTS),
        "subscription": RNG.choice(SUB_TYPES, size=n_users, p=SUB_WEIGHTS),
        "platform": RNG.choice(PLATFORMS, size=n_users, p=PLATFORM_WEIGHTS),
        "skip_tendency": RNG.choice(SKIP_TENDENCY, size=n_users, p=SKIP_WEIGHTS)
    })
    return users


users_df = generate_users(N_USERS)
print(f"[INFO] Generated Users table with shape {users_df.shape}")


# Create Tracks table 
GENRES = ["pop", "hiphop", "electronic", "rock", "latin", "classical", "indie", "jazz"]
GENRE_WEIGHTS = [0.25, 0.20, 0.15, 0.15, 0.10, 0.05, 0.07, 0.03]

def generate_tracks(n_tracks: int):
    """Create synthetic track-level attributes."""
    tracks = pd.DataFrame({
        "track_id": np.arange(1, n_tracks + 1),
        "genre": RNG.choice(GENRES, size=n_tracks, p=GENRE_WEIGHTS),
        "popularity": RNG.integers(0, 101, size=n_tracks),
        "acousticness": RNG.random(size=n_tracks),
        "danceability": RNG.random(size=n_tracks),
        "energy": RNG.random(size=n_tracks),
        "tempo": RNG.integers(60, 181, size=n_tracks),     # bpm 60–180
        "duration_sec": RNG.integers(120, 360, size=n_tracks)  # 2–6 mins
    })
    return tracks


tracks_df = generate_tracks(N_TRACKS)
print(f"[INFO] Generated Tracks table with shape {tracks_df.shape}")


# Create Sessions table 
TIME_OF_DAY = ["morning", "afternoon", "evening", "night"]
TOD_WEIGHTS = [0.20, 0.35, 0.30, 0.15]

DAY_TYPE = ["weekday", "weekend"]
DAY_WEIGHTS = [0.70, 0.30]

LOCATIONS = ["home", "commute", "work", "gym"]
LOC_WEIGHTS = [0.50, 0.25, 0.20, 0.05]

def generate_sessions(n_sessions: int):
    """Create basic session metadata."""
    sessions = pd.DataFrame({
        "session_id": np.arange(1, n_sessions + 1),
        "user_id": RNG.integers(1, N_USERS + 1, size=n_sessions),
        "time_of_day": RNG.choice(TIME_OF_DAY, size=n_sessions, p=TOD_WEIGHTS),
        "day_type": RNG.choice(DAY_TYPE, size=n_sessions, p=DAY_WEIGHTS),
        "location": RNG.choice(LOCATIONS, size=n_sessions, p=LOC_WEIGHTS),
        "session_length": RNG.integers(5, 40, size=n_sessions)  # 5–40 tracks
    })
    return sessions


sessions_df = generate_sessions(N_SESSIONS)
print(f"[INFO] Generated Sessions table with shape {sessions_df.shape}")

# Generate Events table 
def expand_sessions_to_events(sessions_df):
    """
    Expand each session into individual listening events.
    Each event will later receive a track and skip label.
    """
    rows = []
    for _, row in sessions_df.iterrows():
        session_id = row["session_id"]
        n_events = row["session_length"]

        for pos in range(n_events):
            rows.append({
                "session_id": session_id,
                "user_id": row["user_id"],
                "position": pos + 1,
                "time_of_day": row["time_of_day"],
                "day_type": row["day_type"],
                "location": row["location"]
            })

    return pd.DataFrame(rows)


events_df = expand_sessions_to_events(sessions_df)
print(f"[INFO] Expanded to Events table with shape {events_df.shape}")


# Assign tracks 
def assign_tracks(events_df, tracks_df):
    """Randomly assign tracks to each listening event."""
    track_ids = RNG.integers(1, N_TRACKS + 1, size=len(events_df))
    events_df["track_id"] = track_ids
    return events_df


events_df = assign_tracks(events_df, tracks_df)


# Merge Users + Tracks + Sessions 
events_df = events_df.merge(users_df, on="user_id", how="left")
events_df = events_df.merge(tracks_df, on="track_id", how="left")

print(f"[INFO] Merged full Events table with shape {events_df.shape}")


# Skip Probability Model 
def compute_skip_probability(row):
    """Compute skip probability based on several interacting factors."""

    p = 0.25  # baseline skip rate

    # ---- User-level signals ----
    if row["skip_tendency"] == "high":
        p += 0.25
    elif row["skip_tendency"] == "medium":
        p += 0.10

    # Age differences
    if row["age_group"] in ["13-17", "18-24"]:
        p += 0.05

    # Subscription type
    if row["subscription"] == "free":
        p += 0.10

    # Country-level skip culture
    if row["country"] == "LATAM":
        p += 0.08
    if row["country"] == "EU_UK":
        p -= 0.05

    # ----- Session-level signals -----
    if row["time_of_day"] == "morning":
        p -= 0.05
    if row["location"] == "gym":
        p += 0.10
    if row["day_type"] == "weekend":
        p += 0.05

    # ----- Track-level signals -----
    if row["genre"] in ["pop", "latin"]:
        p -= 0.05
    if row["genre"] in ["classical", "jazz"]:
        p += 0.10

    # Song audio features
    if row["energy"] < 0.3:
        p += 0.05
    if row["danceability"] > 0.7:
        p -= 0.05

    # Clamp between 0 and 1
    return max(0, min(1, p))


# Apply the skip probability model
events_df["skip_prob"] = events_df.apply(compute_skip_probability, axis=1)

# Draw Bernoulli outcome
events_df["skip"] = RNG.random(len(events_df)) < events_df["skip_prob"]

print("[INFO] Assigned skip labels.")


# Save Final Dataset 
events_df.to_csv(OUTPUT_FILE, index=False)
print(f"[INFO] Saved final dataset to {OUTPUT_FILE}")