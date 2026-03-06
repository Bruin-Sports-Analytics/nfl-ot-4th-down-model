"""
app.py
======
Streamlit interactive 4th-down conversion probability predictor.

Run with:
    streamlit run src/app.py
"""

import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (project-relative — no hardcoded machine paths)
# ---------------------------------------------------------------------------
_ROOT           = Path(__file__).resolve().parents[1]
MODEL_PATH      = _ROOT / "models" / "fourth_down_conversion.pkl"
TEAM_STATS_PATH = _ROOT / "data" / "processed" / "team_stats_snapshot.csv"

# ---------------------------------------------------------------------------
# Load model and team stats
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"Model not found at `{MODEL_PATH}`.\n\n"
            "Run the training pipeline first:\n"
            "```\n"
            "python src/build_features.py\n"
            "python -m src.models.fourth_down_conversion\n"
            "```"
        )
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_team_stats():
    if not TEAM_STATS_PATH.exists():
        st.error(
            f"Team stats not found at `{TEAM_STATS_PATH}`.\n\n"
            "Run `python src/build_features.py` first."
        )
        st.stop()
    return pd.read_csv(TEAM_STATS_PATH)

model_data = load_model()
cal_model  = model_data["model"]
features   = model_data["features"]
team_stats = load_team_stats()
teams      = sorted(team_stats["team"].dropna().tolist())

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="4th Down Conversion Predictor",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 4th Down Conversion Predictor")
st.markdown(
    "Select teams and enter the game situation to get a calibrated conversion probability. "
    "Team stats reflect rolling 15-game averages."
)
st.caption(
    "Model: XGBoost + isotonic calibration · "
    "Hyperparameters tuned via RandomizedSearchCV · "
    "NFL play-by-play data (2012–2024)"
)
st.divider()

# ---------------------------------------------------------------------------
# Layout: sidebar for team selection, main area for situation + result
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Teams")
    offense = st.selectbox(
        "Offensive Team",
        teams,
        index=teams.index("KC") if "KC" in teams else 0,
    )
    defense = st.selectbox(
        "Defensive Team",
        teams,
        index=teams.index("SF") if "SF" in teams else 1,
    )

    st.divider()
    st.header("Situation")
    ydstogo  = st.slider("Yards To Go",                       min_value=1,    max_value=10,   value=2)
    yardline = st.slider("Yard Line (from opponent end zone)", min_value=1,    max_value=99,   value=35)
    qtr      = st.selectbox("Quarter", [1, 2, 3, 4, "OT"], index=3)
    qtr_val  = 5 if qtr == "OT" else int(qtr)

    st.divider()
    st.header("Clock & Score")
    score_diff     = st.slider("Score Differential (offense − defense)", min_value=-35, max_value=35,   value=0)
    secs_remaining = st.slider("Seconds Remaining in Game",              min_value=0,   max_value=3600, value=300, step=10)
    half_secs      = st.slider("Seconds Remaining in Half",              min_value=0,   max_value=1800, value=min(300, secs_remaining), step=10)

    st.divider()
    st.header("Game State")
    wp_val      = st.slider("Win Probability (offense)", min_value=0.0, max_value=1.0, value=0.45, step=0.01)
    off_to      = st.slider("Offense Timeouts",          min_value=0,   max_value=3,   value=2)
    def_to      = st.slider("Defense Timeouts",          min_value=0,   max_value=3,   value=2)

    st.divider()
    st.header("Environment")
    shotgun_sel = st.selectbox("Formation", ["Shotgun", "Under Center"])
    shotgun_val = 1 if shotgun_sel == "Shotgun" else 0
    no_huddle   = st.checkbox("No-Huddle / Hurry-Up", value=False)
    temp        = st.slider("Temperature (°F)", min_value=20, max_value=105, value=65)

# ---------------------------------------------------------------------------
# Build prediction input
# ---------------------------------------------------------------------------
try:
    off_row = team_stats[team_stats["team"] == offense].iloc[0]
    def_row = team_stats[team_stats["team"] == defense].iloc[0]
except IndexError:
    st.error("Team stats missing for selected team. Re-run build_features.py.")
    st.stop()

def _get(row, col, default=0.0):
    return float(row[col]) if col in row.index and pd.notna(row[col]) else default

input_dict = {
    "ydstogo":                    ydstogo,
    "qtr":                        qtr_val,
    "yardline_100":               yardline,
    "score_differential":         score_diff,
    "game_seconds_remaining":     secs_remaining,
    "half_seconds_remaining":     half_secs,
    "wp":                         wp_val,
    "posteam_timeouts_remaining": off_to,
    "defteam_timeouts_remaining": def_to,
    "temp":                       temp,
    "shotgun":                    shotgun_val,
    "no_huddle":                  int(no_huddle),
    "epa_per_game_roll15":        _get(off_row, "epa_per_game_roll15"),
    "success_rate_roll15":        _get(off_row, "success_rate_roll15",     0.42),
    "points_per_game_roll15":     _get(off_row, "points_per_game_roll15",  23.0),
    "go_conv_rate":               _get(off_row, "go_conv_rate",            0.50),
    "def_epa_per_game_roll15":    _get(def_row, "def_epa_per_game_roll15"),
    "def_success_rate_roll15":    _get(def_row, "def_success_rate_roll15", 0.42),
    "def_points_per_game_roll15": _get(def_row, "def_points_per_game_roll15", 23.0),
    "go_stop_rate":               _get(def_row, "go_stop_rate",            0.50),
}

# Only pass features the model was trained on (in correct order)
available = {k: v for k, v in input_dict.items() if k in features}
input_df  = pd.DataFrame([available])[features]
prob      = cal_model.predict_proba(input_df)[0][1]

# ---------------------------------------------------------------------------
# Main content: result + matchup breakdown
# ---------------------------------------------------------------------------
col_res, col_ctx = st.columns([1, 2])

with col_res:
    if prob >= 0.60:
        color   = "#4ade80"
        verdict = "Likely to Convert"
        bg      = "#052e16"
    elif prob >= 0.45:
        color   = "#f5c518"
        verdict = "Toss-Up"
        bg      = "#1c1707"
    else:
        color   = "#f87171"
        verdict = "Unlikely to Convert"
        bg      = "#1c0707"

    st.markdown(f"""
    <div style="text-align:center; padding: 36px 20px; border-radius: 14px;
                background-color: {bg}; border: 2px solid {color}; margin-bottom: 16px;">
        <div style="color:{color}; font-size: 72px; font-weight: 900; line-height: 1;
                    font-family: monospace;">{prob:.1%}</div>
        <div style="color:{color}; font-size: 18px; font-weight: 700;
                    margin-top: 10px;">Conversion Probability</div>
        <div style="font-size: 15px; color: #9ca3af; margin-top: 6px;">{verdict}</div>
        <div style="font-size: 12px; color: #4b5563; margin-top: 12px;">
            4th &amp; {ydstogo} · {offense} vs {defense}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick ydstogo sensitivity strip
    st.markdown("**Sensitivity to yards to go:**")
    for ytg in range(1, min(11, ydstogo + 5)):
        row2 = available.copy(); row2["ydstogo"] = ytg
        p2   = cal_model.predict_proba(pd.DataFrame([row2])[features])[0][1]
        c2   = "#4ade80" if p2 >= 0.55 else "#f5c518" if p2 >= 0.45 else "#f87171"
        marker = " ◄ current" if ytg == ydstogo else ""
        st.markdown(
            f"<span style='font-family:monospace; color:{c2}'>"
            f"4th & {ytg:2d}: {p2:.1%}{marker}</span>",
            unsafe_allow_html=True,
        )

with col_ctx:
    st.subheader("Matchup Context")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{offense} Offense** *(last {ROLL_WINDOW} games)*")
        st.metric("EPA per Drive",       f"{_get(off_row, 'epa_per_game_roll15'):+.3f}")
        st.metric("Success Rate",        f"{_get(off_row, 'success_rate_roll15', 0.42):.1%}")
        st.metric("Pts Scored / Game",   f"{_get(off_row, 'points_per_game_roll15', 23):,.1f}")
        st.metric("4th-Down Conv Rate",  f"{_get(off_row, 'go_conv_rate', 0.50):.1%}")
        if "yards_per_play_roll15" in team_stats.columns:
            st.metric("Yards per Play", f"{_get(off_row, 'yards_per_play_roll15', 5.5):.2f}")

    with c2:
        st.markdown(f"**{defense} Defense** *(last {ROLL_WINDOW} games allowed)*")
        st.metric("EPA per Drive Allowed",      f"{_get(def_row, 'def_epa_per_game_roll15'):+.3f}")
        st.metric("Success Rate Allowed",       f"{_get(def_row, 'def_success_rate_roll15', 0.42):.1%}")
        st.metric("Pts Allowed / Game",         f"{_get(def_row, 'def_points_per_game_roll15', 23):,.1f}")
        st.metric("4th-Down Stop Rate",         f"{_get(def_row, 'go_stop_rate', 0.50):.1%}")
        if "def_yards_per_play_roll15" in team_stats.columns:
            st.metric("Yards per Play Allowed", f"{_get(def_row, 'def_yards_per_play_roll15', 5.5):.2f}")

    st.divider()

    # Situation at a glance
    st.subheader("Situation")
    s1, s2, s3 = st.columns(3)
    s1.metric("Field Position",      f"{yardline} yds out")
    s1.metric("Quarter",             f"Q{qtr}")
    s2.metric("Score Diff",          f"{score_diff:+d}")
    s2.metric("Time Left (game)",    f"{secs_remaining // 60}:{secs_remaining % 60:02d}")
    s3.metric("Win Probability",     f"{wp_val:.1%}")
    s3.metric("Formation",           shotgun_sel)

st.divider()
st.caption(
    "Predictions are based on rolling team stats and situational inputs only. "
    "The model does not account for individual player quality, injuries, specific play-calling, or weather beyond temperature."
)

ROLL_WINDOW = 15