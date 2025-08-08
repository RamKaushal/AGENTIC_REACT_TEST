
# citi_sci_forecast_app_v4.py
# Streamlit "Forecast Command Centre" — Citibank-inspired + Futuristic Sci‑Fi Chat Interface
# STRICT GATED FLOW + ANCHORED ROWS + New Day/Week graphs + Interactive Calendar
# To run: streamlit run citi_sci_forecast_app_v4.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import altair as alt
import time

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Forecast Command Centre — CitiSphere",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------
# Theme & CSS
# -----------------------
CITI_BLUE = "#004C97"
CITI_RED = "#E4002B"
NEON = "#00E5FF"
BG0 = "#050914"
BG1 = "#081327"

st.markdown(f"""
<style>
:root {{
  --citi-blue: {CITI_BLUE};
  --citi-red: {CITI_RED};
  --neon: {NEON};
  --bg0: {BG0};
  --bg1: {BG1};
}}
.stApp {{
  background:
    radial-gradient(1200px 600px at 15% 0%, rgba(0,229,255,0.09), transparent 60%),
    radial-gradient(1000px 500px at 85% 100%, rgba(228,0,43,0.08), transparent 60%),
    linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
  color: #D9E7FF;
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji";
}}
.brandbar {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  border: 1px solid rgba(0,229,255,0.22);
  border-radius: 14px;
  background: linear-gradient(90deg, rgba(0,76,151,0.55), rgba(10,26,68,0.6));
  box-shadow: 0 0 40px rgba(0,76,151,0.25);
  margin-bottom: 8px;
}}
.brand-badge {{ width: 13px; height: 13px; border-radius: 50%; background: var(--citi-red); box-shadow: 0 0 12px rgba(228,0,43,0.6); }}
.brand-title {{ font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }}
.brand-sub {{ opacity: 0.85; font-size: 0.9rem; margin-left: auto; }}
.right-pane-wrapper {{ border-left: 1px dashed rgba(0,229,255,0.22); margin-left: 0.75rem; padding-left: 0.9rem; }}
.bubble {{
  border: 1px solid rgba(0,229,255,0.35);
  background: linear-gradient(180deg, rgba(0,229,255,0.06), rgba(0,229,255,0.02));
  border-radius: 16px;
  padding: 0.75rem 1rem;
  margin: 0.55rem 0;
  box-shadow: 0 0 18px rgba(0,229,255,0.08), inset 0 0 32px rgba(0,229,255,0.05);
}}
.bubble.system {{ border-color: rgba(228,0,43,0.45); box-shadow: 0 0 18px rgba(228,0,43,0.15), inset 0 0 32px rgba(228,0,43,0.06); }}
.captionline {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.9rem; opacity: 0.9; }}

.stDataFrame, .stTable {{ background: rgba(5,12,28,0.55) !important; border: 1px solid rgba(0,229,255,0.2); border-radius: 12px; }}

.stButton > button {{
  background: linear-gradient(180deg, #0F2960, #0A1C44);
  color: #E6F0FF; border: 1px solid rgba(0,229,255,0.35);
  border-radius: 12px; padding: 0.45rem 0.9rem;
}}
.stButton > button:hover {{ box-shadow: 0 0 16px rgba(0,229,255,0.35); border-color: var(--neon); }}
.stTextInput > div > div > input, .stTextArea textarea, .stSelectbox, .stDateInput input {{
  background: rgba(5, 12, 28, 0.75); color: #D9E7FF; border: 1px solid rgba(0,229,255,0.25); border-radius: 10px;
}}

.stepchip {{ display: inline-flex; align-items: center; gap: 8px; padding: 4px 10px; border: 1px solid rgba(0,229,255,0.3);
  border-radius: 999px; background: rgba(0,229,255,0.06); font-size: 0.85rem; margin-bottom: 6px; }}
.dot-ok {{ width: 8px; height: 8px; border-radius: 50%; background: #13FF9E; box-shadow: 0 0 10px rgba(19,255,158,0.6); }}
.dot-warn {{ width: 8px; height: 8px; border-radius: 50%; background: #FFC857; box-shadow: 0 0 10px rgba(255,200,87,0.6); }}

/* Anchored row spacers to align right outputs with left steps */
.rowspacer {{ border: 0; padding: 0; margin: 0.25rem 0; }}
.rowspacer > div {{ min-height: var(--minh); }}
</style>
""", unsafe_allow_html=True)

full_text = '👋 Hello Ram, Good Morning!!'
display_text = ''

# Container for the title
with st.container():
    text_placeholder = st.empty()  # Placeholder to update text

    # Typewriter animation: show 1 more character each time
    for i in range(len(full_text) + 1):
        display_text = full_text[:i]
        text_placeholder.markdown(
            f"""
            <div class="title-hero">
                <h2 style="margin:0">{display_text}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.08)  # Adjust speed (seconds per character)

st.markdown(
    '<div class="brandbar">'
    '<div class="brand-badge"></div>'
    '<div class="brand-title">Forecast Command Centre — CitiSphere</div>'
    '<div class="brand-sub">Citibank-inspired palette • Sci‑Fi UX</div>'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------
# Helpers & State
# -----------------------
def init_state():
    defaults = {
        # NEW top steps
        "show_day_prompted": False,
        "show_day": False,
        "show_week_prompted": False,
        "show_week": False,

        # Original flow (shifted down)
        "started": False,  # now Step 3
        "base_ran": False,
        "festival_ran": False,
        "base_approved": None,         # True/False
        "base_human_confirmed": False, # when NO path is confirmed
        "human_adj_df": None,

        "season_ran": False,
        "season_approved": None,
        "season_confirmed": False,     # confirm tweak when NO
        "season_df": None,

        "pulse_ran": False,
        "pulse_approved": None,
        "pulse_confirmed": False,      # confirm comment when NO
        "pulse_df": None,

        "add_info_done": False,
        "add_info_text": "",

        "foresight_ran": False,
        "foresight_df": None,
        "foresight_weights": None,

        "exec_summary_toggle": False,
        "exec_summary_text": None,

        "email_toggle": False,
        "email_to": "",
        "email_subject": "Forecast summary",
        "email_sent": False,

        "holidays": None,
        "trend_vals": None
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_from(step: str):
    order = [
        "day","week","start","base","festival","base_approval","season","season_approval",
        "pulse","pulse_approval","add_info","foresight","exec","email"
    ]
    idx = order.index(step)
    if idx <= order.index("week"):
        st.session_state["show_week"] = False
        st.session_state["show_week_prompted"] = False
    if idx <= order.index("start"):
        for k in ["started","base_ran","festival_ran","base_approved","base_human_confirmed","human_adj_df"]:
            st.session_state[k] = False if isinstance(st.session_state[k], bool) else None
    if idx <= order.index("season"):
        for k in ["season_ran","season_approved","season_confirmed","season_df"]:
            st.session_state[k] = False if isinstance(st.session_state[k], bool) else None
    if idx <= order.index("pulse"):
        for k in ["pulse_ran","pulse_approved","pulse_confirmed","pulse_df"]:
            st.session_state[k] = False if isinstance(st.session_state[k], bool) else None
    if idx <= order.index("add_info"):
        for k in ["add_info_done","add_info_text"]:
            st.session_state[k] = False if isinstance(st.session_state[k], bool) else ""
    if idx <= order.index("foresight"):
        for k in ["foresight_ran","foresight_df","foresight_weights"]:
            st.session_state[k] = False if isinstance(st.session_state[k], bool) else None
    if idx <= order.index("exec"):
        for k in ["exec_summary_toggle","exec_summary_text"]:
            st.session_state[k] = False if isinstance(st.session_state[k], bool) else None
    if idx <= order.index("email"):
        for k in ["email_toggle","email_to","email_subject","email_sent"]:
            st.session_state[k] = False if isinstance(st.session_state[k], bool) else ("" if isinstance(st.session_state[k], str) else None)

init_state()

# Data helpers
def make_dates(n=2, start=None):
    start = start or (date.today() + timedelta(days=1))
    return [start + timedelta(days=i) for i in range(n)]

def df_from_dict(d):
    return pd.DataFrame(d)

# Sample numbers (spec)
base_values = [12340, 12880]
festival_values = [11100, 12050]
human_values = [11450, 12300]
season_values = [11450, 12280]
pulse_values = [12900, 13020]
dates = make_dates(2)

spec_base_df = df_from_dict({"Date": dates, "Base Forecast": base_values})
spec_fest_df = df_from_dict({"Date": dates, "Base Forecast": base_values, "Festival Forecast": festival_values})
spec_human_df = df_from_dict({"Date": dates, "Base": base_values, "Festive": festival_values, "Human-Adj Forecast": human_values})
spec_season_df = df_from_dict({"Date": dates, "Seasoncast Adjusted Forecast": season_values})
spec_pulse_df = df_from_dict({"Date": dates, "MTD Forecast": pulse_values})

default_weights = { "Base": 0.25, "Festival": 0.15, "Season": 0.25, "Pulse": 0.25, "Human": 0.10 }

def learn_weights(agent_matrix, actuals):
    try:
        w, *_ = np.linalg.lstsq(agent_matrix, actuals, rcond=None)
        w = np.clip(w, 0, None)
        if w.sum() > 0: w = w / w.sum()
        return w
    except Exception:
        return None

def make_stable_trend():
    base = 1000
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 8, size=42)
    vals = base + np.cumsum(noise)
    return vals

if st.session_state["trend_vals"] is None:
    st.session_state["trend_vals"] = make_stable_trend()

# Mock holiday markers
today = date.today()
this_month_start = today.replace(day=1)
end_of_month = (this_month_start.replace(day=28) + timedelta(days=10)).replace(day=1) - timedelta(days=1)
h1 = this_month_start + timedelta(days=11)  # 12th
h2 = this_month_start + timedelta(days=18)  # 19th
st.session_state["holidays"] = [h1, h2]

# -----------------------
# Day & Week chart data
# -----------------------
def yesterday_series():
    # hourly points for yesterday
    yday = date.today() - timedelta(days=1)
    hours = pd.date_range(pd.Timestamp(yday), periods=24, freq="H")
    base = np.linspace(1000, 1200, 24)
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 15, 24)
    forecast = base
    actual = base + noise
    df = pd.DataFrame({"t": hours, "Forecast": forecast, "Actual": actual})
    return df

def week_series():
    # last 7 days at daily granularity
    days = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=7, freq="D")
    base = np.linspace(6800, 7200, 7)
    rng = np.random.default_rng(9)
    noise = rng.normal(0, 50, 7)
    forecast = base
    actual = base + noise
    df = pd.DataFrame({"t": days, "Forecast": forecast, "Actual": actual})
    return df

def dual_line_chart(df, title):
    df_long = df.melt("t", var_name="Series", value_name="Value")
    # Solid for Forecast, dashed for Actual
    stroke_dash = alt.condition(alt.datum.Series == "Actual", alt.value([4,4]), alt.value([1]))
    color_scale = alt.Scale(domain=["Forecast","Actual"], range=["#6FA8FF","#FFB4C9"])
    chart = (
        alt.Chart(df_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("t:T", title="Time"),
            y=alt.Y("Value:Q", title=None),
            color=alt.Color("Series:N", scale=color_scale),
            strokeDash=stroke_dash,
            tooltip=[alt.Tooltip("t:T", title="Time"), "Series:N", alt.Tooltip("Value:Q", title="Value", format=".2f")]
        )
        .properties(height=220, title=title)
    )
    return chart

# -----------------------
# Anchored right-side row helpers
# -----------------------
def spacer(min_px: int = 80):
    st.markdown(f"""<div class="rowspacer"><div style="--minh:{min_px}px"></div></div>""", unsafe_allow_html=True)

# Suggested min-heights per row to align with left steps (tuned visually)
ROW_H = {
    1: 280,  # Day-level graph
    2: 280,  # Week-level graph
    3: 420,  # Interactive calendar + trend
    4: 140,  # base table
    5: 140,  # festival table
    6: 160,  # final forecast table
    7: 140,  # season header
    8: 160,  # season results
    9: 140,  # pulse header
    10: 140, # pulse results
    11: 120, # add info echo
    12: 200, # foresight results
    13: 180, # executive summary
    14: 120, # email confirmation
}

# -----------------------
# Layout
# -----------------------
left, right = st.columns([0.44, 0.56], gap="small")

# ---------- LEFT: INPUT / CONTROLS
with left:
    st.markdown("### PAGE 1 — Main Chat (Split View) — **Input**")
    # New Step 1
    st.markdown('<div class="bubble">1) Do you want to show yesterday result?</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("YES ✅ (show day graph)", key="show_day_yes"):
        st.session_state["show_day"] = True
        st.session_state["show_day_prompted"] = True
    if c2.button("NO ❌", key="show_day_no"):
        st.session_state["show_day"] = False
        st.session_state["show_day_prompted"] = True
        reset_from("day")

    # New Step 2 (only appears after Step 1 YES, per gating rule)
    if st.session_state["show_day"]:
        st.markdown('<div class="bubble">2) Do you want to see 7 days summary?</div>', unsafe_allow_html=True)
        c21, c22 = st.columns(2)
        if c21.button("YES ✅ (show week graph)", key="show_week_yes"):
            st.session_state["show_week"] = True
            st.session_state["show_week_prompted"] = True
        if c22.button("NO ❌", key="show_week_no"):
            st.session_state["show_week"] = False
            st.session_state["show_week_prompted"] = True
            reset_from("week")

    # Original flow now starts here as Step 3
    if st.session_state["show_day_prompted"] and (st.session_state["show_day"] or st.session_state["show_week_prompted"]):
        st.markdown('<div class="bubble">3) Do you want to start Forecast?</div>', unsafe_allow_html=True)
        c3a, c3b = st.columns(2)
        if c3a.button("YES ✅", key="start_yes"):
            st.session_state["started"] = True
        if c3b.button("NO ❌", key="start_no"):
            st.session_state["started"] = False
            reset_from("start")

    if st.session_state["started"]:
        st.markdown('<div class="bubble">4) Agent Base Forecast Triggered</div>', unsafe_allow_html=True)
        if st.button("Run Base Forecast ▶️", key="run_base"):
            st.session_state["base_ran"] = True
            st.session_state["festival_ran"] = False
            st.session_state["base_approved"] = None
            st.session_state["base_human_confirmed"] = False
            st.session_state["human_adj_df"] = None

        if st.session_state["base_ran"]:
            st.markdown('<div class="bubble">5) Agent Festivalcast Triggered</div>', unsafe_allow_html=True)
            if st.button("Apply Festival Adjustments 🎉", key="run_fest"):
                st.session_state["festival_ran"] = True
                st.session_state["base_approved"] = None
                st.session_state["base_human_confirmed"] = False

        if st.session_state["festival_ran"]:
            st.markdown('<div class="bubble">6) Do you approve the forecast?</div>', unsafe_allow_html=True)
            c4, c5 = st.columns(2)
            if c4.button("YES ✅ (Base/Fest)", key="approve_base"):
                st.session_state["base_approved"] = True
                st.session_state["human_adj_df"] = spec_human_df.copy()
            if c5.button("NO, add human adj ✍️", key="reject_base"):
                st.session_state["base_approved"] = False
            if st.session_state["base_approved"] is False:
                hadj = []
                for i, dt in enumerate(dates):
                    v = st.number_input(f"Human adjustment for {dt.isoformat()}",
                                        min_value=0, value=human_values[i], step=10, key=f"hadj_{i}")
                    hadj.append(v)
                if st.button("Confirm human adjustments ✅", key="confirm_hadj"):
                    st.session_state["human_adj_df"] = df_from_dict({
                        "Date": dates, "Base": base_values, "Festive": festival_values,
                        "Human-Adj Forecast": hadj
                    })
                    st.session_state["base_human_confirmed"] = True

    can_go_season = (st.session_state["base_approved"] is True) or st.session_state["base_human_confirmed"]
    if can_go_season:
        st.markdown('<div class="bubble">7) Seasoncast is triggered</div>', unsafe_allow_html=True)
        if st.button("Run Seasoncast 📈", key="run_season"):
            st.session_state["season_ran"] = True
            st.session_state["season_approved"] = None
            st.session_state["season_confirmed"] = False

    if st.session_state["season_ran"]:
        st.markdown('<div class="bubble">8) Do you approve the seasonal f’cast?</div>', unsafe_allow_html=True)
        c6, c7 = st.columns(2)
        if c6.button("YES ✅ (Season)", key="approve_season"):
            st.session_state["season_approved"] = True
            st.session_state["season_df"] = spec_season_df.copy()
        if c7.button("NO, tweak ✍️", key="reject_season"):
            st.session_state["season_approved"] = False
        if st.session_state["season_approved"] is False:
            s_adj = []
            for i, dt in enumerate(dates):
                s_val = st.number_input(f"Seasonal adj for {dt.isoformat()}",
                                        min_value=0, value=season_values[i], step=10, key=f"sadj_{i}")
                s_adj.append(s_val)
            if st.button("Confirm seasonal tweak ✅", key="confirm_sadj"):
                st.session_state["season_df"] = df_from_dict({
                    "Date": dates, "Seasoncast Adjusted Forecast": s_adj
                })
                st.session_state["season_confirmed"] = True

    can_go_pulse = (st.session_state["season_approved"] is True) or st.session_state["season_confirmed"]
    if can_go_pulse:
        st.markdown('<div class="bubble">9) Pulsecast is triggered</div>', unsafe_allow_html=True)
        if st.button("Fetch MTD (Pulse) 🛰️", key="run_pulse"):
            st.session_state["pulse_ran"] = True
            st.session_state["pulse_approved"] = None
            st.session_state["pulse_confirmed"] = False

    if st.session_state["pulse_ran"]:
        st.markdown('<div class="bubble">10) Do you approve the Pulse f’cast?</div>', unsafe_allow_html=True)
        c8, c9 = st.columns(2)
        if c8.button("YES ✅ (Pulse)", key="approve_pulse"):
            st.session_state["pulse_approved"] = True
            st.session_state["pulse_df"] = spec_pulse_df.copy()
        if c9.button("NO, comment ✍️", key="reject_pulse"):
            st.session_state["pulse_approved"] = False
        if st.session_state["pulse_approved"] is False:
            st.text_input("Enter feedback / instruction for Pulsecast", key="pulse_comment")
            if st.button("Confirm Pulse note ✅", key="confirm_pulse_note"):
                st.session_state["pulse_df"] = spec_pulse_df.copy()
                st.session_state["pulse_confirmed"] = True

    can_add_info = (st.session_state["pulse_approved"] is True) or st.session_state["pulse_confirmed"]
    if can_add_info:
        st.markdown("---")
        st.markdown('<div class="bubble">11) Add any additional info before Foresight?</div>', unsafe_allow_html=True)
        c10, c11 = st.columns(2)
        if c10.button("YES, add context 🧩", key="add_info_yes"):
            st.session_state["add_info_done"] = True
        if c11.button("NO", key="add_info_no"):
            st.session_state["add_info_done"] = True  # proceed
        if st.session_state["add_info_done"]:
            st.session_state["add_info_text"] = st.text_area("Notes / context / events (optional)", height=100, key="addinfo")
            st.markdown('<div class="bubble">12) Trigger Foresight</div>', unsafe_allow_html=True)
            if st.button("Run Foresight 🤖➕", key="run_foresight"):
                b = np.array(base_values, dtype=float)
                f = np.array(festival_values, dtype=float)
                s = np.array(season_values, dtype=float) if st.session_state["season_df"] is None else \
                    np.array(st.session_state["season_df"]["Seasoncast Adjusted Forecast"].values, dtype=float)
                p = np.array(pulse_values, dtype=float)
                h = np.array(st.session_state["human_adj_df"]["Human-Adj Forecast"].values, dtype=float) \
                    if st.session_state["human_adj_df"] is not None else (b + f + s + p) / 4.0

                A = np.vstack([b, f, s, p, h]).T
                rng = np.random.default_rng(7)
                actuals = h * rng.normal(1.0, 0.01, size=h.shape)
                w = learn_weights(A, actuals)
                if w is None:
                    weights = np.array([default_weights[k] for k in ["Base","Festival","Season","Pulse","Human"]])
                else:
                    weights = w

                foresight = (A * weights).sum(axis=1)

                st.session_state["foresight_df"] = pd.DataFrame({
                    "Date": dates, "Base": b.astype(int), "Festiv": f.astype(int), "Season": s.astype(int),
                    "Pulse": p.astype(int), "Human": h.astype(int), "Final F’c": foresight.astype(int)
                })
                st.session_state["foresight_weights"] = dict(zip(["Base","Festival","Season","Pulse","Human"], np.round(weights, 3)))
                st.session_state["foresight_ran"] = True

            if st.session_state["foresight_ran"]:
                st.markdown('<div class="bubble">13) Generate executive summary?</div>', unsafe_allow_html=True)
                c12, c13 = st.columns(2)
                if c12.button("YES, generate 📝", key="exec_yes"):
                    w = st.session_state["foresight_weights"] or default_weights
                    add = st.session_state["add_info_text"]
                    summary = (
                        f"**Why the forecast moved**: Blended weights leaned on Base/Season/Pulse, "
                        f"with weights {w}. Festival dampened peaks near holidays. "
                        f"Human inputs provided fine-grained guardrails.\n\n"
                        f"**Insights**: Recent 6‑week trend is stable; minor pre‑holiday softening followed by post‑holiday reversion. "
                        f"Fraud signal flagged; volatility risk moderate. \n\n"
                        f"**Recommendations**: Maintain guardrails ±3%, tighten anomaly detection around holidays, "
                        f"and revisit Pulse ingestion cadence (daily AM)."
                    )
                    if add and add.strip():
                        summary += f"\n\n**User Context Added**: {add.strip()}"
                    st.session_state["exec_summary_text"] = summary
                    st.session_state["exec_summary_toggle"] = True
                if c13.button("NO", key="exec_no"):
                    st.session_state["exec_summary_toggle"] = False

                st.markdown('<div class="bubble">14) Email the report?</div>', unsafe_allow_html=True)
                c14, c15 = st.columns(2)
                if c14.button("YES, email 📧", key="email_yes"):
                    st.session_state["email_toggle"] = True
                if c15.button("NO", key="email_no"):
                    st.session_state["email_toggle"] = False
                if st.session_state["email_toggle"]:
                    st.session_state["email_to"] = st.text_input("To:", value=st.session_state["email_to"])
                    st.session_state["email_subject"] = st.text_input("Subject:", value=st.session_state["email_subject"])
                    if st.button("Send (simulate)", key="send_email"):
                        st.session_state["email_sent"] = True

# ---------- RIGHT: OUTPUT / CHAT VIEW (Anchored Rows)
with right:
    st.markdown("### Output")
    st.markdown('<div class="right-pane-wrapper">', unsafe_allow_html=True)

    # Pre-create 14 row containers
    r = [st.container() for _ in range(14)]

    # Row 1 — Day-level graph (yesterday)
    with r[0]:
        if st.session_state["show_day"]:
            st.markdown('<div class="bubble system captionline">DAY-LEVEL GRAPH (Yesterday)</div>', unsafe_allow_html=True)
            day_df = yesterday_series()
            st.altair_chart(dual_line_chart(day_df, "Yesterday — Forecast (solid) vs Actual (dashed)"), use_container_width=True)
        else:
            spacer(ROW_H[1])

    # Row 2 — Week-level graph (last 7 days) only after Step 1 YES + Step 2 YES
    with r[1]:
        if st.session_state["show_day"] and st.session_state["show_week"]:
            st.markdown('<div class="bubble system captionline">WEEK-LEVEL GRAPH (Last 7 Days)</div>', unsafe_allow_html=True)
            wk_df = week_series()
            st.altair_chart(dual_line_chart(wk_df, "Last 7 Days — Forecast (solid) vs Actual (dashed)"), use_container_width=True)
        else:
            spacer(ROW_H[2])

    # Row 3 — Calendar + Trends (interactive calendar)
    with r[2]:
        if st.session_state["started"]:
            st.markdown('<div class="bubble system captionline">CALENDAR — Interactive (Holidays listed)</div>', unsafe_allow_html=True)
            # interactive calendar for current month
            picked = st.date_input("Select date or range", value=(this_month_start, end_of_month))
            h1, h2 = st.session_state["holidays"]
            st.markdown(f"◼ Holiday on **{h1.strftime('%b %d')}**  ◼ Holiday on **{h2.strftime('%b %d')}**")
            st.markdown('<div class="bubble system captionline">SEASONALITY TRENDS (Last 6 Weeks)</div>', unsafe_allow_html=True)
            trend = pd.DataFrame({"t": pd.date_range(end=pd.Timestamp.today(), periods=len(st.session_state['trend_vals'])),
                                "value": st.session_state["trend_vals"]})
            st.line_chart(trend.set_index("t"))
            st.markdown('<div class="bubble captionline">I’m thinking about holidays present in the range…</div>', unsafe_allow_html=True)
            st.markdown('<div class="bubble captionline">I’m thinking about seasonality before holidays… (last 6 weeks data consistent)</div>', unsafe_allow_html=True)
        else:
            spacer(ROW_H[3])

    # Row 4 — Base table
    with r[3]:
        if st.session_state["base_ran"]:
            st.markdown('<div class="bubble system captionline">BASE FORECAST TABLE</div>', unsafe_allow_html=True)
            st.dataframe(spec_base_df, use_container_width=True)
        else:
            spacer(ROW_H[4])

    # Row 5 — Festival table
    with r[4]:
        if st.session_state["festival_ran"]:
            st.markdown('<div class="bubble system captionline">ADJUSTED (FESTIVAL) TABLE</div>', unsafe_allow_html=True)
            st.dataframe(spec_fest_df, use_container_width=True)
        else:
            spacer(ROW_H[5])

    # Row 6 — Final forecast after approval or confirmed human adj
    with r[5]:
        if (st.session_state["base_approved"] is True) or st.session_state["base_human_confirmed"]:
            st.markdown('<div class="bubble system captionline">FINAL FORECAST</div>', unsafe_allow_html=True)
            df_show = st.session_state["human_adj_df"] if st.session_state["human_adj_df"] is not None else spec_human_df
            st.dataframe(df_show, use_container_width=True)
        else:
            spacer(ROW_H[6])

    # Row 7 — Season header/info
    with r[6]:
        if (st.session_state["base_approved"] is True) or st.session_state["base_human_confirmed"]:
            st.markdown('<div class="bubble system captionline">SEASONCAST INSIGHTS</div>', unsafe_allow_html=True)
            st.markdown('<div class="stepchip"><div class="dot-warn"></div> Warning: fraud signal noted; trend may change</div>', unsafe_allow_html=True)
        else:
            spacer(ROW_H[7])

    # Row 8 — Season results
    with r[7]:
        if (st.session_state["season_approved"] is True) or st.session_state["season_confirmed"]:
            st.dataframe(st.session_state["season_df"] if st.session_state["season_df"] is not None else spec_season_df, use_container_width=True)
        else:
            spacer(ROW_H[8])

    # Row 9 — Pulse header
    with r[8]:
        if (st.session_state["season_approved"] is True) or st.session_state["season_confirmed"]:
            st.markdown('<div class="bubble system captionline">PULSECAST (MTD)</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown("> [ Outlook email screenshot / summary placeholder ]")
        else:
            spacer(ROW_H[9])

    # Row 10 — Pulse results
    with r[9]:
        if (st.session_state["pulse_approved"] is True) or st.session_state["pulse_confirmed"]:
            st.dataframe(st.session_state["pulse_df"] if st.session_state["pulse_df"] is not None else spec_pulse_df, use_container_width=True)
        else:
            spacer(ROW_H[10])

    # Row 11 — Add-info echo (optional)
    with r[10]:
        if st.session_state["add_info_done"]:
            if st.session_state["add_info_text"].strip():
                st.markdown(f"**User Context Added**: {st.session_state['add_info_text'].strip()}")
        else:
            spacer(ROW_H[11])

    # Row 12 — Foresight results
    with r[11]:
        if st.session_state["foresight_ran"] and st.session_state["foresight_df"] is not None:
            st.markdown('<div class="bubble system captionline">FORESIGHT THOUGHT & ACTIONS (LOG)</div>', unsafe_allow_html=True)
            st.markdown("Thought: adjust forecasts of Base, Festival, Season, Pulse, and Human (if any).  \nAction: learned weights from last 6 weeks + agent f’casts.")
            st.dataframe(st.session_state["foresight_df"], use_container_width=True)
            if st.session_state["foresight_weights"]:
                weights_str = "  ".join([f"{k}={v:.2f}" for k,v in st.session_state["foresight_weights"].items()])
                st.markdown(f"**Weights used (learned)**: {weights_str}")
        else:
            spacer(ROW_H[12])

    # Row 13 — Exec summary
    with r[12]:
        if st.session_state["exec_summary_toggle"] and st.session_state["exec_summary_text"]:
            st.markdown('<div class="bubble system captionline">EXECUTIVE SUMMARY</div>', unsafe_allow_html=True)
            st.markdown(st.session_state["exec_summary_text"])
        else:
            spacer(ROW_H[13])

    # Row 14 — Email confirmation
    with r[13]:
        if st.session_state["email_sent"]:
            st.success(f"Email sent to {st.session_state['email_to']!s} with subject “{st.session_state['email_subject']!s}”. (Simulated)")
        else:
            spacer(ROW_H[14])

    st.markdown('</div>', unsafe_allow_html=True)
