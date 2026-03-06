"""
app.py
======
Streamlit interactive 4th down conversion probability predictor.

Run with:
    python -m streamlit run src/app.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# Paths
# =============================================================================
BASE_DIR        = Path(r"C:\Users\andre\OneDrive\Desktop\BSA 4th Down Model")
MODEL_PATH      = BASE_DIR / "data" / "conversion_model.pkl"
TEAM_STATS_PATH = BASE_DIR / "data" / "team_stats_snapshot.csv"

# =============================================================================
# Load model and team stats
# =============================================================================
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_team_stats():
    return pd.read_csv(TEAM_STATS_PATH)

model_data = load_model()
cal_model  = model_data["model"]
features   = model_data["features"]
team_stats = load_team_stats()
teams      = sorted(team_stats["team"].tolist())

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="4th Down Conversion Predictor",
    page_icon="🏈",
    layout="centered",
)

st.title("🏈 4th Down Conversion Predictor")
st.markdown("Select teams and input the game situation to get a conversion probability estimate.")
st.caption("Model trained on NFL play-by-play data (2016–2021). Team stats reflect rolling 15-game averages.")

st.divider()

# =============================================================================
# Team selection
# =============================================================================
col1, col2 = st.columns(2)
with col1:
    offense = st.selectbox("Offensive Team", teams, index=teams.index("KC") if "KC" in teams else 0)
with col2:
    defense = st.selectbox("Defensive Team", teams, index=teams.index("SF") if "SF" in teams else 1)

st.divider()

# =============================================================================
# Situation inputs
# =============================================================================
col3, col4 = st.columns(2)
with col3:
    ydstogo  = st.slider("Yards To Go", min_value=1, max_value=10, value=2)
    yardline = st.slider("Yard Line (distance from end zone)", min_value=1, max_value=99, value=35)
with col4:
    score_diff     = st.slider("Score Differential (offense - defense)", min_value=-35, max_value=35, value=0)
    secs_remaining = st.slider("Seconds Remaining in Game", min_value=0, max_value=3600, value=1800, step=30)

col5, col6 = st.columns(2)
with col5:
    shotgun     = st.selectbox("Formation", ["Under Center", "Shotgun"])
    shotgun_val = 1 if shotgun == "Shotgun" else 0
with col6:
    temp = st.slider("Temperature (°F)", min_value=20, max_value=100, value=65)

st.divider()

# =============================================================================
# Build prediction input
# =============================================================================
off_row = team_stats[team_stats["team"] == offense].iloc[0]
def_row = team_stats[team_stats["team"] == defense].iloc[0]

input_dict = {
    "ydstogo":                    ydstogo,
    "yardline_100":               yardline,
    "score_differential":         score_diff,
    "game_seconds_remaining":     secs_remaining,
    "temp":                       temp,
    "shotgun":                    shotgun_val,
    "epa_per_game_roll15":        off_row["epa_per_game_roll15"],
    "success_rate_roll15":        off_row["success_rate_roll15"],
    "points_per_game_roll15":     off_row["points_per_game_roll15"],
    "go_conv_rate":               off_row["go_conv_rate"],
    "go_stop_rate":               off_row["go_stop_rate"],
    "def_epa_per_game_roll15":    def_row["def_epa_per_game_roll15"],
    "def_success_rate_roll15":    def_row["def_success_rate_roll15"],
    "def_points_per_game_roll15": def_row["def_points_per_game_roll15"],
}

input_df = pd.DataFrame([input_dict])[features]
prob = cal_model.predict_proba(input_df)[0][1]

# =============================================================================
# Display result
# =============================================================================
if prob >= 0.60:
    color   = "#16a34a"
    verdict = "Likely to Convert"
elif prob >= 0.45:
    color   = "#d97706"
    verdict = "Toss-Up"
else:
    color   = "#dc2626"
    verdict = "Unlikely to Convert"

st.markdown(f"""
<div style="text-align:center; padding: 30px; border-radius: 12px;
            background-color: {color}22; border: 2px solid {color};">
    <h1 style="color:{color}; font-size: 64px; margin: 0;">{prob:.1%}</h1>
    <h3 style="color:{color}; margin-top: 8px;">Conversion Probability</h3>
    <p style="font-size: 18px; color: #374151;">{verdict}</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# =============================================================================
# Matchup breakdown
# =============================================================================
st.subheader("Matchup Context")

col7, col8 = st.columns(2)

with col7:
    st.markdown(f"**{offense} Offense (last 15 games)**")
    st.metric("EPA per Drive",              f"{off_row['epa_per_game_roll15']:.3f}")
    st.metric("Success Rate",               f"{off_row['success_rate_roll15']:.1%}")
    st.metric("Pts Scored per Game",        f"{off_row['points_per_game_roll15']:.1f}")
    st.metric("4th Down Conv Rate",         f"{off_row['go_conv_rate']:.1%}")

with col8:
    st.markdown(f"**{defense} Defense (last 15 games allowed)**")
    st.metric("EPA per Drive Allowed",      f"{def_row['def_epa_per_game_roll15']:.3f}")
    st.metric("Success Rate Allowed",       f"{def_row['def_success_rate_roll15']:.1%}")
    st.metric("Pts Allowed per Game",       f"{def_row['def_points_per_game_roll15']:.1f}")

st.divider()
st.caption("Predictions are based on team rolling averages and game situation only. Model does not account for individual player quality, injuries, or specific scheme matchups.")
