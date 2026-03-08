"""
app.py
======
Streamlit interactive 4th-down conversion probability predictor.

Run with:
    streamlit run app.py
"""

import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROLL_WINDOW = 15

# ---------------------------------------------------------------------------
# Paths (project-relative — no hardcoded machine paths)
# ---------------------------------------------------------------------------
_ROOT           = Path(__file__).resolve().parent
MODEL_PATH      = _ROOT / "models" / "fourth_down_conversion.pkl"
TEAM_STATS_PATH = _ROOT / "data" / "processed" / "team_stats_snapshot.csv"

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="4th Down Conversion Predictor",
    page_icon="🏈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0a1628; }
    .stMetric { background-color: #0f2219; border-radius: 8px; padding: 8px; }
    .metric-card {
        background: linear-gradient(135deg, #0f2219 0%, #0d1f17 100%);
        border: 1px solid #2d5a3d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .stSelectbox > div > div { background-color: #0f2219; }
</style>
""", unsafe_allow_html=True)

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
            "python src/data/build_features.py\n"
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
            "Run `python src/data/build_features.py` first."
        )
        st.stop()
    return pd.read_csv(TEAM_STATS_PATH)

model_data = load_model()
cal_model  = model_data["model"]
features   = model_data["features"]
team_stats = load_team_stats()
teams      = sorted(team_stats["team"].dropna().tolist())

st.title("🏈 4th Down Conversion Predictor")
st.markdown(
    "Select teams and enter the game situation to get a calibrated conversion probability. "
    "Team stats reflect rolling 15-game averages."
)
st.caption(
    "Model: XGBoost + isotonic calibration · "
    "Hyperparameters tuned via RandomizedSearchCV · "
    "NFL play-by-play data (2016–2024)"
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
    st.error("Team stats missing for selected team. Re-run src/data/build_features.py.")
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
    "def_epa_per_game_roll15":    _get(def_row, "def_epa_per_game_roll15"),
    "def_success_rate_roll15":    _get(def_row, "def_success_rate_roll15", 0.42),
    "def_points_per_game_roll15": _get(def_row, "def_points_per_game_roll15", 23.0),
}

# Only pass features the model was trained on (in correct order)
available = {k: v for k, v in input_dict.items() if k in features}
input_df  = pd.DataFrame([available])[features]
prob      = cal_model.predict_proba(input_df)[0][1]

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def prob_color(p):
    if p >= 0.58:
        return "#4ade80", "#052e16"
    elif p >= 0.42:
        return "#f5c518", "#1c1707"
    else:
        return "#f87171", "#1c0707"

color, bg = prob_color(prob)
verdict = "Likely to Convert" if prob >= 0.58 else "Toss-Up" if prob >= 0.42 else "Unlikely to Convert"

# ---------------------------------------------------------------------------
# Row 1: Gauge + Sensitivity strip + Quick stats
# ---------------------------------------------------------------------------
col_gauge, col_sens, col_stats = st.columns([1, 1, 2])

with col_gauge:
    # Plotly gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge",
        value=prob * 100,
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8aaa96",
                     "tickfont": {"color": "#8aaa96"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#0f2219",
            "borderwidth": 2,
            "bordercolor": "#2d5a3d",
            "steps": [
                {"range": [0, 42],  "color": "#1c0707"},
                {"range": [42, 58], "color": "#1c1707"},
                {"range": [58, 100],"color": "#052e16"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 3},
                "thickness": 0.85,
                "value": 50,
            },
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#0d1f17",
        font={"color": "#f8f8f2", "family": "monospace"},
        margin=dict(t=20, b=0, l=30, r=30),
        height=200,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    delta = prob * 100 - 50
    delta_str = f"{delta:+.1f}% vs 50%"
    st.markdown(
        f"<div style='text-align:center; font-family:monospace;'>"
        f"<span style='font-size:52px; font-weight:bold; color:{color};'>{prob*100:.1f}%</span><br>"
        f"<span style='font-size:13px; color:#8aaa96;'>{delta_str}</span><br>"
        f"<span style='font-size:15px; color:{color};'>{verdict}</span><br>"
        f"<span style='font-size:12px; color:#8aaa96;'>4th & {ydstogo} · {offense} vs {defense}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_sens:
    st.markdown("**Yards-to-Go Sensitivity**")
    ytg_range  = list(range(1, 11))
    ytg_probs  = []
    for ytg in ytg_range:
        row2 = available.copy(); row2["ydstogo"] = ytg
        p2 = cal_model.predict_proba(pd.DataFrame([row2])[features])[0][1]
        ytg_probs.append(p2 * 100)

    colors_bar = [prob_color(p / 100)[0] for p in ytg_probs]
    fig_bar = go.Figure(go.Bar(
        x=ytg_range, y=ytg_probs,
        marker_color=colors_bar,
        marker_line_color="#2d5a3d",
        marker_line_width=1,
        text=[f"{p:.0f}%" for p in ytg_probs],
        textposition="outside",
        textfont={"size": 10, "color": "#f8f8f2"},
    ))
    fig_bar.add_vline(x=ydstogo, line_dash="dash", line_color="#ffffff", line_width=2,
                      annotation_text="current", annotation_font_color="#f8f8f2",
                      annotation_font_size=10)
    fig_bar.update_layout(
        paper_bgcolor="#0d1f17",
        plot_bgcolor="#0f2219",
        font={"color": "#f8f8f2", "family": "monospace"},
        margin=dict(t=20, b=40, l=10, r=10),
        height=280,
        xaxis=dict(title="Yards To Go", tickvals=ytg_range,
                   gridcolor="#1e4a2e", color="#8aaa96"),
        yaxis=dict(title="Conv %", range=[0, 110],
                   gridcolor="#1e4a2e", color="#8aaa96"),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_stats:
    st.markdown("**Matchup Snapshot**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{offense} Offense**")
        st.metric("EPA/Drive",        f"{_get(off_row, 'epa_per_game_roll15'):+.3f}")
        st.metric("Success Rate",     f"{_get(off_row, 'success_rate_roll15', 0.42):.1%}")
        st.metric("Pts/Game",         f"{_get(off_row, 'points_per_game_roll15', 23):,.1f}")
        st.metric("4th Conv Rate",    f"{_get(off_row, 'go_conv_rate', 0.50):.1%}")
    with c2:
        st.markdown(f"**{defense} Defense**")
        st.metric("EPA/Drive Allowed",   f"{_get(def_row, 'def_epa_per_game_roll15'):+.3f}")
        st.metric("Success Rate Allowed",f"{_get(def_row, 'def_success_rate_roll15', 0.42):.1%}")
        st.metric("Pts Allowed/Game",    f"{_get(def_row, 'def_points_per_game_roll15', 23):,.1f}")
        st.metric("4th Stop Rate",       f"{_get(def_row, 'go_stop_rate', 0.50):.1%}")

st.divider()

# ---------------------------------------------------------------------------
# Row 2: Field Position Heatmap + Score/Time Sensitivity
# ---------------------------------------------------------------------------
col_heat, col_time = st.columns([3, 2])

with col_heat:
    st.subheader("Conversion Probability Heatmap")
    st.caption("By yards-to-go and field position (using current team stats)")

    ytg_vals  = list(range(1, 11))
    yard_vals = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 99]
    heat_data = np.zeros((len(ytg_vals), len(yard_vals)))

    for i, ytg in enumerate(ytg_vals):
        for j, yl in enumerate(yard_vals):
            row_h = available.copy()
            row_h["ydstogo"]     = ytg
            row_h["yardline_100"] = yl
            p_h = cal_model.predict_proba(pd.DataFrame([row_h])[features])[0][1]
            heat_data[i, j] = p_h * 100

    fig_heat = go.Figure(go.Heatmap(
        z=heat_data,
        x=[f"{y} yds" for y in yard_vals],
        y=[f"& {ytg}" for ytg in ytg_vals],
        colorscale=[
            [0.0, "#1c0707"], [0.35, "#f87171"],
            [0.50, "#f5c518"],
            [0.65, "#4ade80"], [1.0, "#052e16"],
        ],
        zmin=0, zmax=100,
        text=np.round(heat_data, 0).astype(int),
        texttemplate="%{text}%",
        textfont={"size": 9, "color": "#ffffff"},
        colorbar=dict(
            title="Conv %", ticksuffix="%",
            tickfont={"color": "#8aaa96"},
            title_font={"color": "#8aaa96"},
        ),
        showscale=True,
    ))
    # Highlight current cell
    curr_j = min(range(len(yard_vals)), key=lambda j: abs(yard_vals[j] - yardline))
    curr_i = ydstogo - 1
    fig_heat.add_shape(
        type="rect",
        x0=curr_j - 0.5, x1=curr_j + 0.5,
        y0=curr_i - 0.5, y1=curr_i + 0.5,
        line=dict(color="#ffffff", width=3),
    )
    fig_heat.update_layout(
        paper_bgcolor="#0d1f17",
        plot_bgcolor="#0f2219",
        font={"color": "#f8f8f2", "family": "monospace"},
        margin=dict(t=20, b=60, l=60, r=20),
        height=350,
        xaxis=dict(title="Distance from Opponent End Zone", color="#8aaa96",
                   tickangle=-30),
        yaxis=dict(title="4th Down &...", color="#8aaa96", autorange="reversed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_time:
    st.subheader("Score & Time Sensitivity")
    st.caption("How conversion probability shifts with score and clock")

    # Score differential sweep
    score_range = list(range(-21, 22, 3))
    score_probs = []
    for sd in score_range:
        row_s = available.copy(); row_s["score_differential"] = sd
        p_s = cal_model.predict_proba(pd.DataFrame([row_s])[features])[0][1]
        score_probs.append(p_s * 100)

    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(
        x=score_range, y=score_probs,
        mode="lines+markers",
        line=dict(color="#f5c518", width=2.5),
        marker=dict(size=6, color="#f5c518"),
        name="Conv %",
        fill="tozeroy",
        fillcolor="rgba(245, 197, 24, 0.08)",
    ))
    fig_score.add_vline(x=score_diff, line_dash="dash", line_color="#ffffff", line_width=1.5,
                        annotation_text=f"{score_diff:+d}", annotation_font_color="#ffffff",
                        annotation_font_size=10)
    fig_score.add_hline(y=50, line_dash="dot", line_color="#8aaa96", line_width=1)
    fig_score.update_layout(
        paper_bgcolor="#0d1f17", plot_bgcolor="#0f2219",
        font={"color": "#f8f8f2", "family": "monospace"},
        margin=dict(t=10, b=40, l=10, r=10),
        height=150,
        xaxis=dict(title="Score Diff (off − def)", color="#8aaa96", gridcolor="#1e4a2e", zeroline=True, zerolinecolor="#2d5a3d"),
        yaxis=dict(title="Conv %", range=[0, 100], gridcolor="#1e4a2e", color="#8aaa96"),
        showlegend=False,
    )
    st.plotly_chart(fig_score, use_container_width=True)

    # Time remaining sweep (last 15 min of game)
    time_range = list(range(0, 901, 60))
    time_probs = []
    for t in time_range:
        row_t = available.copy(); row_t["game_seconds_remaining"] = t
        p_t = cal_model.predict_proba(pd.DataFrame([row_t])[features])[0][1]
        time_probs.append(p_t * 100)

    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=time_range, y=time_probs,
        mode="lines+markers",
        line=dict(color="#4ade80", width=2.5),
        marker=dict(size=6, color="#4ade80"),
        name="Conv %",
        fill="tozeroy",
        fillcolor="rgba(74, 222, 128, 0.08)",
    ))
    if secs_remaining <= 900:
        fig_time.add_vline(x=secs_remaining, line_dash="dash", line_color="#ffffff", line_width=1.5,
                           annotation_text=f"{secs_remaining//60}:{secs_remaining%60:02d}",
                           annotation_font_color="#ffffff", annotation_font_size=10)
    fig_time.add_hline(y=50, line_dash="dot", line_color="#8aaa96", line_width=1)
    fig_time.update_layout(
        paper_bgcolor="#0d1f17", plot_bgcolor="#0f2219",
        font={"color": "#f8f8f2", "family": "monospace"},
        margin=dict(t=10, b=40, l=10, r=10),
        height=150,
        xaxis=dict(title="Seconds Remaining (last 15 min)", color="#8aaa96",
                   gridcolor="#1e4a2e",
                   tickvals=[0, 180, 360, 540, 720, 900],
                   ticktext=["0:00", "3:00", "6:00", "9:00", "12:00", "15:00"]),
        yaxis=dict(title="Conv %", range=[0, 100], gridcolor="#1e4a2e", color="#8aaa96"),
        showlegend=False,
    )
    st.plotly_chart(fig_time, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Row 3: All-team comparison bar chart
# ---------------------------------------------------------------------------
st.subheader(f"How All Teams Would Fare — 4th & {ydstogo} from the {yardline}-yd line")
st.caption("Using each team's current rolling stats as the offense, with selected defense")

team_probs = []
for team in teams:
    try:
        t_row = team_stats[team_stats["team"] == team].iloc[0]
        t_dict = {
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
            "epa_per_game_roll15":        _get(t_row, "epa_per_game_roll15"),
            "success_rate_roll15":        _get(t_row, "success_rate_roll15",     0.42),
            "points_per_game_roll15":     _get(t_row, "points_per_game_roll15",  23.0),
            "def_epa_per_game_roll15":    _get(def_row, "def_epa_per_game_roll15"),
            "def_success_rate_roll15":    _get(def_row, "def_success_rate_roll15", 0.42),
            "def_points_per_game_roll15": _get(def_row, "def_points_per_game_roll15", 23.0),
        }
        t_avail = {k: v for k, v in t_dict.items() if k in features}
        t_df    = pd.DataFrame([t_avail])[features]
        p_t     = cal_model.predict_proba(t_df)[0][1]
        team_probs.append({"team": team, "prob": p_t * 100})
    except Exception:
        pass

df_teams = pd.DataFrame(team_probs).sort_values("prob", ascending=True)
bar_colors_teams = [prob_color(p / 100)[0] for p in df_teams["prob"]]
# Highlight selected offense
highlight = ["#ffffff" if t == offense else c
             for t, c in zip(df_teams["team"], bar_colors_teams)]

fig_teams = go.Figure(go.Bar(
    y=df_teams["team"],
    x=df_teams["prob"],
    orientation="h",
    marker_color=bar_colors_teams,
    marker_line_color=["#ffffff" if t == offense else "#1e4a2e" for t in df_teams["team"]],
    marker_line_width=[2.5 if t == offense else 0.5 for t in df_teams["team"]],
    text=[f"{p:.1f}%" for p in df_teams["prob"]],
    textposition="outside",
    textfont={"size": 10, "color": "#f8f8f2"},
))
fig_teams.add_vline(x=50, line_dash="dot", line_color="#8aaa96", line_width=1.5)
fig_teams.update_layout(
    paper_bgcolor="#0d1f17", plot_bgcolor="#0f2219",
    font={"color": "#f8f8f2", "family": "monospace"},
    margin=dict(t=10, b=40, l=60, r=80),
    height=700,
    xaxis=dict(title="Conversion Probability (%)", range=[0, 115],
               gridcolor="#1e4a2e", color="#8aaa96"),
    yaxis=dict(color="#8aaa96", tickfont={"size": 11}),
    showlegend=False,
)
st.plotly_chart(fig_teams, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Situation at a glance
# ---------------------------------------------------------------------------
st.subheader("Situation Summary")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Field Position",   f"{yardline} yds out")
s1.metric("Quarter",          f"Q{qtr}")
s2.metric("Score Diff",       f"{score_diff:+d}")
s2.metric("Time Left (game)", f"{secs_remaining // 60}:{secs_remaining % 60:02d}")
s3.metric("Win Probability",  f"{wp_val:.1%}")
s3.metric("Formation",        shotgun_sel)
s4.metric("Off Timeouts",     str(off_to))
s4.metric("Def Timeouts",     str(def_to))

st.divider()
st.caption(
    "Predictions are based on rolling team stats and situational inputs only. "
    "The model does not account for individual player quality, injuries, specific play-calling, or weather beyond temperature. "
    "Data: NFL play-by-play 2016–2024 via nflfastR."
)
