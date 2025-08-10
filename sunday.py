# citi_sci_forecast_app_v5_fix.py
# Row-paired Streamlit layout with strict "next-left only after previous-right" gating

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import altair as alt
import time
import plotly.graph_objects as go

st.set_page_config(page_title="Forecast Command Centre — CitiSphere", layout="wide", initial_sidebar_state="collapsed")

# ---------- THEME/CSS ----------
CITI_BLUE = "#004C97"; CITI_RED = "#E4002B"; NEON = "#00E5FF"; BG0 = "#050914"; BG1 = "#081327"
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
  font-family: "Inter","Segoe UI",system-ui,-apple-system,Roboto,"Helvetica Neue",Arial;
}}
.brandbar {{
  display:flex; align-items:center; gap:10px; padding:10px 16px;
  border:1px solid rgba(0,229,255,0.22); border-radius:14px;
  background:linear-gradient(90deg, rgba(0,76,151,0.55), rgba(10,26,68,0.6));
  box-shadow:0 0 40px rgba(0,76,151,0.25); margin:8px 0 6px 0;
}}
.brand-badge {{ width:13px; height:13px; border-radius:50%; background: var(--citi-red); box-shadow:0 0 12px rgba(228,0,43,0.6); }}
.brand-title {{ font-weight:800; letter-spacing:.08em; text-transform:uppercase; }}
.brand-sub {{ opacity:.85; font-size:.9rem; margin-left:auto; }}
.bubble {{
  border:1px solid rgba(0,229,255,0.35);
  background:linear-gradient(180deg, rgba(0,229,255,0.06), rgba(0,229,255,0.02));
  border-radius:16px; padding:.75rem 1rem; margin:.55rem 0;
  box-shadow:0 0 18px rgba(0,229,255,0.08), inset 0 0 32px rgba(0,229,255,0.05);
}}
.bubble.system {{ border-color:rgba(228,0,43,0.45); box-shadow:0 0 18px rgba(228,0,43,0.15), inset 0 0 32px rgba(0,228,255,0.06); }}
.captionline {{ font-family: ui-monospace, Menlo, Consolas, "Courier New", monospace; font-size:.9rem; opacity:.9; }}
.rowwrap {{ border-left:1px dashed rgba(0,229,255,0.22); padding-left:.9rem; margin-left:.5rem; }}
.stDataFrame,.stTable {{ background:rgba(5,12,28,0.55)!important; border:1px solid rgba(0,229,255,0.2); border-radius:12px; }}
.stButton>button {{
  background:linear-gradient(180deg,#0F2960,#0A1C44); color:#E6F0FF; border:1px solid rgba(0,229,255,0.35);
  border-radius:12px; padding:.45rem .9rem;
}}
.stButton>button:hover {{ box-shadow:0 0 16px rgba(0,229,255,0.35); border-color: var(--neon); }}
.stTextInput>div>div>input, .stTextArea textarea, .stDateInput input {{
  background:rgba(5,12,28,0.75); color:#D9E7FF; border:1px solid rgba(0,229,255,0.25); border-radius:10px;
}}
.stepchip {{ display:inline-flex; align-items:center; gap:8px; padding:4px 10px; border:1px solid rgba(0,229,255,0.3);
  border-radius:999px; background:rgba(0,229,255,0.06); font-size:.85rem; margin-bottom:6px; }}
.dot-warn {{ width:8px; height:8px; border-radius:50%; background:#FFC857; box-shadow:0 0 10px rgba(255,200,87,0.6); }}
.row-title {{ font-weight:700; opacity:.9; margin-bottom:.2rem; }}
</style>
""", unsafe_allow_html=True)


# Hero
full_text='👋 Hello Ram, Good Morning!!'; text_placeholder=st.empty()
for i in range(len(full_text)+1):
    text_placeholder.markdown(f'<h2 style="margin:0">{full_text[:i]}</h2>', unsafe_allow_html=True)
    time.sleep(0.01)

st.markdown('<div class="brandbar"><div class="brand-badge"></div><div class="brand-title">Forecast Command Centre — CitiSphere</div><div class="brand-sub">Citibank-inspired palette • Sci-Fi UX</div></div>', unsafe_allow_html=True)

# ---------- STATE ----------
def init_state():
    d={
        "show_day":False, "show_day_done":False,          # step 1
        "show_week":False, "show_week_done":False,        # step 2 (must be YES to proceed)
        "started":False,                                  # step 3
        "base_ran":False,                                 # step 4
        "festival_ran":False,                             # step 5
        "base_approved":None, "base_human_confirmed":False, "human_adj_df":None,  # step 6
        "season_ran":False,                               # step 7
        "season_approved":None, "season_confirmed":False, "season_df":None,"season_raw_df": None,       # step 8
        "pulse_ran":False,                                # step 9
        "pulse_approved":None, "pulse_confirmed":False, "pulse_df":None,          # step10
        "add_info_done":False, "add_info_text":"",        # step11
        "foresight_ran":False, "foresight_df":None, "foresight_weights":None,     # step12
        "exec_summary_toggle":False, "exec_summary_text":None,                    # step13
        "email_toggle":False, "email_to":"", "email_subject":"Forecast summary", "email_sent":False # step14
    }
    for k,v in d.items():
        if k not in st.session_state: st.session_state[k]=v
init_state()

# ---------- DATA ----------
def make_dates(n=2,start=None):
    start = start or (date.today()+timedelta(days=1)); return [start+timedelta(days=i) for i in range(n)]
def df_from_dict(d): return pd.DataFrame(d)
base_values=[12340,12880]; festival_values=[11100,12050]; human_values=[11450,12300]; season_values=[11450,12280]; pulse_values=[12900,13020]
dates=make_dates(2)
spec_base_df=df_from_dict({"Date":dates,"Base Forecast":base_values})
spec_fest_df=df_from_dict({"Date":dates,"Base Forecast":base_values,"Festival Forecast":festival_values})
spec_human_df=df_from_dict({"Date":dates,"Base":base_values,"Festive":festival_values,"Human-Adj Forecast":human_values})
spec_season_df=df_from_dict({"Date":dates,"Seasoncast Adjusted Forecast":season_values})
spec_pulse_df=df_from_dict({"Date":dates,"MTD Forecast":pulse_values})

def learn_weights(A,y):
    try:
        w,*_=np.linalg.lstsq(A,y,rcond=None); w=np.clip(w,0,None); w=w/w.sum() if w.sum()>0 else w; return w
    except Exception: return None

# Charts

def volume_fig(df, title="Yesterday — Volume (Actual vs Forecast)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Time"], y=df["Forecast Volume"], name="Forecast Volume",
        mode="lines", line=dict(width=3, dash="dash"),
        hovertemplate="Time: %{x|%H:%M}<br>Forecast: %{y:,}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df["Time"], y=df["Actual Volume"], name="Actual Volume",
        mode="lines+markers", line=dict(width=3),
        hovertemplate="Time: %{x|%H:%M}<br>Actual: %{y:,}<extra></extra>"
    ))
    fig.update_layout(
        title=title, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(title=None, tickformat="%H:%M", dtick=7_200_000),  # 2 hours in ms
        yaxis=dict(title="Volume", tickformat=",d")
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikemode="across")
    return fig

def aht_fig(df, title="Yesterday — AHT (Actual vs Forecast)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Time"], y=df["Forecast AHT"], name="Forecast AHT",
        mode="lines", line=dict(width=3, dash="dash"),
        hovertemplate="Time: %{x|%H:%M}<br>Forecast AHT: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df["Time"], y=df["Actual AHT"], name="Actual AHT",
        mode="lines+markers", line=dict(width=3),
        hovertemplate="Time: %{x|%H:%M}<br>Actual AHT: %{y}<extra></extra>"
    ))
    fig.update_layout(
        title=title, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(title=None, tickformat="%H:%M", dtick=7_200_000),  # 2 hours
        yaxis=dict(title="AHT (seconds)")
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikemode="across")
    return fig
def yesterday_series():
    import numpy as np
    import pandas as pd
    from datetime import date, timedelta

    # Time: yesterday, 30-min intervals (48 rows)
    yday = date.today() - timedelta(days=1)
    times = pd.date_range(pd.Timestamp(yday), periods=48, freq="30min")

    rng = np.random.default_rng(42)

    # Smooth daily shape to vary volume through the day
    x = np.linspace(0, 2*np.pi, 48, endpoint=False)
    shape = np.sin(x - np.pi/2) + 0.3*np.sin(2*x) + 1.3
    shape = (shape - shape.min()) / (shape.max() - shape.min())  # normalize 0..1

    # Volume in 19k..30k (forecast + noisy actual)
    v_min, v_max = 19_000, 30_000
    forecast_vol = v_min + (v_max - v_min) * shape
    vol_noise = rng.normal(0, (v_max - v_min) * 0.03, size=48)   # ~3% noise
    actual_vol = np.clip(forecast_vol + vol_noise, v_min, v_max)

    # AHT in 450..500 (slight inverse relation to volume + small noise)
    aht_base = 475 + 10*np.cos(x)                                # ~465..485
    aht_noise_f = rng.normal(0, 2.0, size=48)                    # tiny forecast jitter
    aht_noise_a = rng.normal(0, 4.0, size=48)                    # slightly larger actual jitter
    forecast_aht = np.clip(aht_base + aht_noise_f, 450, 500)
    actual_aht   = np.clip(aht_base + aht_noise_a, 450, 500)

    # SLA (%) ~60..90, lower when load/AHT are higher
    aht_norm = (actual_aht - 450) / 50.0                         # normalize AHT 0..1
    sla_noise = rng.normal(0.0, 0.02, size=48)                   # ~±2 percentage points
    sla = 0.90 - 0.15*shape - 0.10*aht_norm + sla_noise
    sla = np.clip(sla, 0.60, 0.90)
    sla_percent = np.round(sla * 100, 1)

    return pd.DataFrame({
        "Time": times,
        "Actual Volume": np.rint(actual_vol).astype(int),
        "Actual AHT": np.rint(actual_aht).astype(int),
        "Forecast Volume": np.rint(forecast_vol).astype(int),
        "Forecast AHT": np.rint(forecast_aht).astype(int),
        "SLA": sla_percent,   # percent, e.g., 83.2
    })

########
import numpy as np
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go

def yesterday_series_2(n_days=7, seed=42):
    """
    Build last n_days (ending yesterday) at DAILY level.
    Columns: Date, Actual Volume, Forecast Volume, Actual AHT, Forecast AHT, SLA
    AHT & SLA are volume-weighted from 30-min intraday simulation.
    """
    rng = np.random.default_rng(seed)

    # 30-min timeline across last n_days (ending yesterday)
    start = date.today() - timedelta(days=n_days)
    end   = date.today() - timedelta(days=1)
    times = pd.date_range(pd.Timestamp(start),
                          pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=30),
                          freq="30min")

    # Intraday shape reused each day
    x = np.linspace(0, 2*np.pi, 48, endpoint=False)
    shape = np.sin(x - np.pi/2) + 0.3*np.sin(2*x) + 1.3
    shape = (shape - shape.min()) / (shape.max() - shape.min())
    shape = np.tile(shape, len(pd.date_range(start, end, freq="D")))

    # Day-level variation
    n_days_exact = len(pd.date_range(start, end, freq="D"))
    day_scale = rng.normal(1.0, 0.06, size=n_days_exact)   # volume swing per day
    aht_shift = rng.normal(0.0, 6.0, size=n_days_exact)    # AHT shift per day (sec)
    day_scale_48 = np.repeat(day_scale, 48)
    aht_shift_48 = np.repeat(aht_shift, 48)

    # Volume per 30-min
    v_min, v_max = 19_000, 30_000
    base_vol = v_min + (v_max - v_min) * shape
    forecast_vol = base_vol * day_scale_48
    vol_noise = rng.normal(0, (v_max - v_min) * 0.03, size=forecast_vol.size)
    actual_vol = np.clip(forecast_vol + vol_noise, v_min, v_max)

    # AHT per 30-min
    aht_base = 475 + 10*np.cos(np.tile(x, n_days_exact)) + aht_shift_48
    aht_noise_f = rng.normal(0, 2.0, size=aht_base.size)
    aht_noise_a = rng.normal(0, 4.0, size=aht_base.size)
    forecast_aht = np.clip(aht_base + aht_noise_f, 450, 500)
    actual_aht   = np.clip(aht_base + aht_noise_a, 450, 500)

    # SLA per 30-min (percentage), lower when load/AHT higher
    load_norm = (base_vol - v_min) / (v_max - v_min)
    aht_norm = (actual_aht - 450) / 50.0
    sla = 0.90 - 0.15*load_norm - 0.10*aht_norm + rng.normal(0.0, 0.02, size=aht_base.size)
    sla = np.clip(sla, 0.60, 0.90) * 100.0

    # 30-min DF
    df_30 = pd.DataFrame({
        "Time": times,
        "Actual Volume": np.rint(actual_vol).astype(int),
        "Forecast Volume": np.rint(forecast_vol).astype(int),
        "Actual AHT": np.rint(actual_aht).astype(int),
        "Forecast AHT": np.rint(forecast_aht).astype(int),
        "SLA": np.round(sla, 1)
    })

    # Aggregate to daily (volume-weighted AHT & SLA)
    def wavg(series, weights):
        wsum = weights.sum()
        return float((series * weights).sum() / wsum) if wsum else float("nan")

    g = df_30.set_index("Time").resample("D")
    df_day = g.apply(lambda x: pd.Series({
        "Actual Volume":   int(x["Actual Volume"].sum()),
        "Forecast Volume": int(x["Forecast Volume"].sum()),
        "Actual AHT":      round(wavg(x["Actual AHT"], x["Actual Volume"]), 1),
        "Forecast AHT":    round(wavg(x["Forecast AHT"], x["Forecast Volume"]), 1),
        "SLA":             round(wavg(x["SLA"], x["Actual Volume"]), 1),
    })).reset_index().rename(columns={"Time": "Date"})

    return df_day.sort_values("Date").reset_index(drop=True)

def volume_fig_2(df, title="Last 7 Days — Volume (Actual vs Forecast)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Forecast Volume"], name="Forecast Volume",
        mode="lines", line=dict(width=3, dash="dash"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: %{y:,}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Actual Volume"], name="Actual Volume",
        mode="lines+markers", line=dict(width=3),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Actual: %{y:,}<extra></extra>"
    ))
    fig.update_layout(
        title=title, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(title=None, tickformat="%b %d", dtick="D1"),
        yaxis=dict(title="Volume", tickformat=",d")
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikemode="across")
    return fig

def aht_fig_2(df, title="Last 7 Days — AHT (Actual vs Forecast)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Forecast AHT"], name="Forecast AHT",
        mode="lines", line=dict(width=3, dash="dash"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast AHT: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Actual AHT"], name="Actual AHT",
        mode="lines+markers", line=dict(width=3),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Actual AHT: %{y}<extra></extra>"
    ))
    fig.update_layout(
        title=title, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(title=None, tickformat="%b %d", dtick="D1"),
        yaxis=dict(title="AHT (seconds)")
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikemode="across")
    return fig

##############
def week_series():
    days=pd.date_range(end=pd.Timestamp.today().normalize()-pd.Timedelta(days=1), periods=7, freq="D")
    base=np.linspace(6800,7200,7); rng=np.random.default_rng(9); noise=rng.normal(0,50,7)
    return pd.DataFrame({"t":days,"Forecast":base,"Actual":base+noise})
def dual_line_chart(df,title):
    df_long=df.melt("t", var_name="Series", value_name="Value")
    stroke_dash=alt.condition(alt.datum.Series=="Actual", alt.value([4,4]), alt.value([1]))
    color_scale=alt.Scale(domain=["Forecast","Actual"], range=["#6FA8FF","#FFB4C9"])
    return alt.Chart(df_long).mark_line(point=True).encode(
        x=alt.X("t:T", title="Time"), y=alt.Y("Value:Q", title=None),
        color=alt.Color("Series:N", scale=color_scale), strokeDash=stroke_dash,
        tooltip=[alt.Tooltip("t:T", title="Time"), "Series:N", alt.Tooltip("Value:Q", title="Value", format=".2f")]
    ).properties(height=220, title=title)
import streamlit as st
import pandas as pd
import calendar
from datetime import date, datetime
import holidays

def render_holiday_calendar(
    start_date,
    end_date,
    country: str = "IN",
    firstweekday: int = 0,     # 0=Monday, 6=Sunday
    show_legend: bool = True,
    show_holiday_list: bool = True,
    title: str | None = None   # e.g., '<div class="bubble system captionline">CALENDAR</div>'
):
    """
    Render calendars for all months between start_date and end_date (inclusive),
    highlighting public holidays, weekends, and today.

    Parameters
    ----------
    start_date : date | str
        Range start (date object or 'YYYY-MM-DD').
    end_date   : date | str
        Range end (date object or 'YYYY-MM-DD').
    country    : str
        ISO 3166-1 alpha-2 country code for the `holidays` package (e.g., 'IN', 'US').
    firstweekday : int
        0=Monday ... 6=Sunday for the calendar layout.
    show_legend : bool
        Show a small legend under each month.
    show_holiday_list : bool
        Show an expandable list of holidays that fall within the range.
    title : str | None
        Optional HTML string to render as a heading (lets you reuse your .bubble CSS).
    """

    # --- normalize inputs ---
    def to_date(d):
        if isinstance(d, date):
            return d
        if isinstance(d, str):
            return datetime.strptime(d, "%Y-%m-%d").date()
        raise TypeError("start_date/end_date must be date or 'YYYY-MM-DD' string")

    start_date = to_date(start_date)
    end_date = to_date(end_date)

    if start_date > end_date:
        st.warning("Start date is after end date. Please adjust the range.")
        return

    if title:
        st.markdown(title, unsafe_allow_html=True)

    # Month boundaries
    start_month = date(start_date.year, start_date.month, 1)
    end_month = date(end_date.year, end_date.month, 1)

    # Gather years to fetch holidays once
    years = list(range(start_month.year, end_month.year + 1))

    # Get holidays for the span
    try:
        hol = holidays.country_holidays(country, years=years)
    except Exception:
        # Compatibility fallback for older `holidays` versions
        country_cls = getattr(holidays, country, None)
        hol = country_cls(years=years) if country_cls else holidays.HolidayBase()

    # Only keep holiday dates inside the exact range
    holiday_dates = {d for d in hol if start_date <= d <= end_date}
    holiday_names = {d: hol[d] for d in holiday_dates}

    cal = calendar.Calendar(firstweekday=firstweekday)
    today = date.today()

    # Iterate months in the range
    cur_year, cur_month = start_month.year, start_month.month
    while (cur_year, cur_month) <= (end_month.year, end_month.month):
        weeks = cal.monthdayscalendar(cur_year, cur_month)  # 0 for padding cells
        # Mask out days outside [start_date, end_date] (so partial months look clean)
        masked_weeks = []
        for w in weeks:
            row = []
            for d in w:
                if d == 0:
                    row.append("")
                else:
                    the_day = date(cur_year, cur_month, d)
                    if not (start_date <= the_day <= end_date):
                        row.append("")  # hide out-of-range days
                    else:
                        row.append(d)
            masked_weeks.append(row)

        df = pd.DataFrame(masked_weeks, columns=list(calendar.day_abbr)).replace(0, "")

        # --- formatting and styling ---
        def fmt_cell(v):
            if v == "" or v is None:
                return ""
            d = date(cur_year, cur_month, int(v))
            return f"{v} ★" if d in holiday_dates else f"{v}"

        def style_cell(v):
            if v == "" or v is None:
                return ""
            # Value might be "15" or "15 ★" -> extract the number
            try:
                day_num = int(str(v).split()[0])
            except Exception:
                return ""
            d = date(cur_year, cur_month, day_num)
            css = []

            # holidays
            if d in holiday_dates:
                css.append("background-color:rgba(253, 224, 71, 0.15); border:1px solid rgba(253, 224, 71, 0.45);")
            # nice padding/centering
            css.append("text-align:center; padding:6px; border-radius:8px;")
            return "".join(css)

        styled = df.style.format(fmt_cell).applymap(style_cell)

        # --- render month ---
        st.subheader(f"{calendar.month_name[cur_month]} {cur_year}")
        st.table(styled)
        if show_legend:
            st.caption("★ holiday   ·   weekends shaded   ·   today outlined")

        # next month
        if cur_month == 12:
            cur_month, cur_year = 1, cur_year + 1
        else:
            cur_month += 1

    # Holiday list for the whole range
    if show_holiday_list and holiday_names:
        with st.expander("Holiday list in range"):
            for d, name in sorted(holiday_names.items()):
                st.write(f"{d:%a, %b %d, %Y}: {name}")

def row(): return st.container()

# ---------- ROW 1: Yesterday (must press YES to proceed) ----------
with row():
    lcol, rcol = st.columns([0.44,0.56], gap="small")
    with lcol:
        st.markdown('<div class="bubble">1) Do you want to show yesterday result?</div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        if c1.button("YES ✅ (show day graph)", key="show_day_yes"):
            st.session_state["show_day"]=True; st.session_state["show_day_done"]=True
        if c2.button("NO ❌", key="show_day_no"):
            # strict gating: cannot proceed if NO
            st.session_state["show_day"]=False; st.session_state["show_day_done"]=False
    with rcol:
        if st.session_state["show_day"]:
            st.markdown('<div class="bubble system captionline">DAY-LEVEL GRAPH (Yesterday)</div>', unsafe_allow_html=True)
            df = yesterday_series()
            st.plotly_chart(volume_fig(df), use_container_width=True)
            st.plotly_chart(aht_fig(df), use_container_width=True)
            df["SLA"] = pd.to_numeric(df["SLA"], errors="coerce")
            misses = df[df["SLA"] < 70].copy()
            if misses.shape[0]>=1:
               st.warning(f"We’re missing the SLA in time slots {misses['Time'].tolist()}")


           
# ---------- ROW 2: Week (only after Row1 right is visible and YES here) ----------
if st.session_state["show_day"]:
    with row():
        lcol, rcol = st.columns([0.44,0.56], gap="small")
        with lcol:
            st.markdown('<div class="bubble">2) Do you want to see 7 days summary?</div>', unsafe_allow_html=True)
            c21,c22 = st.columns(2)
            if c21.button("YES ✅ (show week graph)", key="show_week_yes"):
                st.session_state["show_week"]=True; st.session_state["show_week_done"]=True
            if c22.button("NO ❌", key="show_week_no"):
                st.session_state["show_week"]=False; st.session_state["show_week_done"]=False
        with rcol:
            if st.session_state["show_week"]:
                st.markdown('<div class="bubble system captionline">WEEK-LEVEL GRAPH (Last 7 Days)</div>', unsafe_allow_html=True)
                df2 = yesterday_series_2()
                st.plotly_chart(volume_fig_2(df2), use_container_width=True)
                st.plotly_chart(aht_fig_2(df2), use_container_width=True)
                df2["SLA"] = pd.to_numeric(df2["SLA"], errors="coerce")
                misses = df2[df2["SLA"] < 80].copy()
                if misses.shape[0]>=1:
                  st.warning(f"We’re missing the SLA in time slots {misses['Date'].tolist()}")

# ---------- ROW 3: Start Forecast (only after Row2 right is visible) ----------
if st.session_state["show_week"]:
    with row():
        lcol, rcol = st.columns([0.44,0.56], gap="small")
        with lcol:
            st.markdown('<div class="bubble">3) Do you want to start Forecast?</div>', unsafe_allow_html=True)
            c3a,c3b = st.columns(2)
            if c3a.button("YES ✅", key="start_yes"):
                st.session_state["started"]=True
            if c3b.button("NO ❌", key="start_no"):
                st.session_state["started"]=False
        with rcol:
            if st.session_state["started"]:
                # Interactive calendar + trend
                t = date.today(); a=t.replace(day=1); b=(a.replace(day=28)+timedelta(days=10)).replace(day=1)-timedelta(days=1)
                start_date, end_date = st.date_input("Select date or range", value=(a,b))


# ---------- ROW 4: Base Forecast (only after Row3 right is visible) ----------
if st.session_state["started"]:
    with row():
        lcol, rcol = st.columns([0.44,0.56], gap="small")
        with lcol:
            st.markdown('<div class="bubble">4) Agent Base Forecast Triggered</div>', unsafe_allow_html=True)
            if st.button("Run Base Forecast ▶️", key="run_base"):
                st.session_state["base_ran"]=True
                st.session_state["festival_ran"]=False
                st.session_state["base_approved"]=None
                st.session_state["base_human_confirmed"]=False
                st.session_state["human_adj_df"]=None
        with rcol:
            if st.session_state["base_ran"]:
                st.markdown('<div class="bubble system captionline">BASE FORECAST TABLE</div>', unsafe_allow_html=True)
                st.dataframe(spec_base_df, use_container_width=True)
                st.markdown('<div class="bubble captionline">Events: There is a marketing event that is present for this as per the Outlook email</div>', unsafe_allow_html=True)
                render_holiday_calendar(start_date, end_date, country="US", firstweekday=0, title=None)
                
		

# ---------- ROW 5: Festivalcast (only after Row4 right is visible) ----------
if st.session_state["base_ran"]:
    with row():
        lcol, rcol = st.columns([0.44,0.56], gap="small")
        with lcol:
            st.markdown('<div class="bubble">5) Agent Festivalcast Triggered</div>', unsafe_allow_html=True)
            if st.button("Apply Festival Adjustments 🎉", key="run_fest"):
                st.session_state["festival_ran"]=True
                st.session_state["base_approved"]=None
                st.session_state["base_human_confirmed"]=False
        with rcol:
            if st.session_state["festival_ran"]:
                st.markdown('<div class="bubble system captionline">ADJUSTED (FESTIVAL) TABLE</div>', unsafe_allow_html=True)
                st.dataframe(spec_fest_df, use_container_width=True)

# ---------- ROW 6: Approve Base/Fest (only after Row5 right is visible) ----------
if st.session_state["festival_ran"]:
    with row():
        lcol, rcol = st.columns([0.44,0.56], gap="small")
        with lcol:
            st.markdown('<div class="bubble">6) Do you approve the forecast?</div>', unsafe_allow_html=True)
            c4,c5 = st.columns(2)
            if c4.button("YES ✅ (Base/Fest)", key="approve_base"):
                st.session_state["base_approved"]=True; st.session_state["human_adj_df"]=spec_human_df.copy()
            if c5.button("NO, add human adj ✍️", key="reject_base"):
                st.session_state["base_approved"]=False
            if st.session_state["base_approved"] is False:
                hadj=[]
                for i,dt in enumerate(dates):
                    v=st.number_input(f"Human adjustment for {dt.isoformat()}", min_value=0, value=human_values[i], step=10, key=f"hadj_{i}")
                    hadj.append(v)
                if st.button("Confirm human adjustments ✅", key="confirm_hadj"):
                    st.session_state["human_adj_df"]=df_from_dict({"Date":dates,"Base":base_values,"Festive":festival_values,"Human-Adj Forecast":hadj})
                    st.session_state["base_human_confirmed"]=True
        with rcol:
            if (st.session_state["base_approved"] is True) or st.session_state["base_human_confirmed"]:
                st.markdown('<div class="bubble system captionline">FINAL FORECAST</div>', unsafe_allow_html=True)
                df_show = st.session_state["human_adj_df"] if st.session_state["human_adj_df"] is not None else spec_human_df
                st.dataframe(df_show, use_container_width=True)
                available_series = ["Base", "Festive"] + (["Human-Adj Forecast"] if "Human-Adj Forecast" in df_show.columns else [])
                default_selection = available_series  # show everything available by default
                selection = st.multiselect("Series to display", available_series, default_selection)

                # --- Build Plotly figure ---
                fig = go.Figure()
                for col in selection:
                    fig.add_trace(
                        go.Scatter(
                            x=df_show["Date"],
                            y=df_show[col],
                            mode="lines+markers",
                            name=col,
                            hovertemplate="<b>%{fullData.name}</b><br>Date=%{x|%Y-%m-%d}<br>Value=%{y:.2f}<extra></extra>",
                        )
                    )

                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="Date",
                    yaxis_title="Value",
                    xaxis=dict(rangeslider=dict(visible=True), type="date"),
                    template="plotly_white",
                    legend_title_text="Series",
                )

                st.plotly_chart(fig, use_container_width=True)


# ---------- ROW 7: Seasoncast Trigger (only after Row6 right is visible) ----------
can_go_season = (st.session_state["base_approved"] is True) or st.session_state["base_human_confirmed"]
if can_go_season:
    with row():
        lcol, rcol = st.columns([0.44,0.56], gap="small")
        with lcol:
            st.markdown('<div class="bubble">7) Seasoncast is triggered</div>', unsafe_allow_html=True)
            if st.button("Run Seasoncast 📈", key="run_season"):
                st.session_state["season_ran"]=True; st.session_state["season_approved"]=None; st.session_state["season_confirmed"]=False
                st.session_state["season_raw_df"] = spec_season_df.copy()   # <-- raw snapshot
                st.session_state["season_df"] = spec_season_df.copy()  
        with rcol:
            if st.session_state["season_ran"]:
                st.markdown('<div class="bubble system captionline">ADJUSTED (SEASON) TABLE</div>', unsafe_allow_html=True)
                st.dataframe(
                    st.session_state.get("season_raw_df", spec_season_df),  # <-- raw, like fest row
                    use_container_width=True
                )
# ---------- ROW 8: Approve Season (only after Row7 right is visible) ----------
if st.session_state["season_ran"]:
    with row():
        lcol, rcol = st.columns([0.44,0.56], gap="small")
        with lcol:
            st.markdown('<div class="bubble">8) Do you approve the seasonal f’cast?</div>', unsafe_allow_html=True)
            c6,c7 = st.columns(2)
            if c6.button("YES ✅ (Season)", key="approve_season"):
                st.session_state["season_approved"]=True; st.session_state["season_df"]=spec_season_df.copy()
            if c7.button("NO, tweak ✍️", key="reject_season"):
                st.session_state["season_approved"]=False
            if st.session_state["season_approved"] is False:
                s_adj=[]
                for i,dt in enumerate(dates):
                    s_val=st.number_input(f"Seasonal adj for {dt.isoformat()}", min_value=0, value=season_values[i], step=10, key=f"sadj_{i}")
                    s_adj.append(s_val)
                if st.button("Confirm seasonal tweak ✅", key="confirm_sadj"):
                    st.session_state["season_df"]=df_from_dict({"Date":dates,"Seasoncast Adjusted Forecast":s_adj}); st.session_state["season_confirmed"]=True
        with rcol:
            if (st.session_state["season_approved"] is True) or st.session_state["season_confirmed"]:
                st.dataframe(st.session_state["season_df"] if st.session_state["season_df"] is not None else spec_season_df, use_container_width=True)
                available_series = ["Date", "Seasoncast Adjusted Forecast"] + (["Human-Adj Forecast"] if "Human-Adj Forecast" in df_show.columns else [])
                default_selection = available_series  # show everything available by default
                selection = st.multiselect("Series to display", available_series, default_selection)

                # --- Build Plotly figure ---
                fig = go.Figure()
                for col in selection:
                    fig.add_trace(
                        go.Scatter(
                            x=st.session_state["season_df"]["Date"],
                            y=st.session_state["season_df"][col],
                            mode="lines+markers",
                            name=col,
                            hovertemplate="<b>%{fullData.name}</b><br>Date=%{x|%Y-%m-%d}<br>Value=%{y:.2f}<extra></extra>",
                        )
                    )

                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="Date",
                    yaxis_title="Value",
                    xaxis=dict(rangeslider=dict(visible=True), type="date"),
                    template="plotly_white",
                    legend_title_text="Series",
                )

                st.plotly_chart(fig, use_container_width=True)

# ---------- ROW 9: Pulsecast Trigger (only after Row8 right is visible) ----------
can_go_pulse = (st.session_state["season_approved"] is True) or st.session_state["season_confirmed"]
if can_go_pulse:
    with row():
        lcol, rcol = st.columns([0.44,0.56], gap="small")
        with lcol:
            st.markdown('<div class="bubble">9) Pulsecast is triggered</div>', unsafe_allow_html=True)
            if st.button("Fetch MTD (Pulse) 🛰️", key="run_pulse"):
                st.session_state["pulse_ran"]=True; st.session_state["pulse_approved"]=None; st.session_state["pulse_confirmed"]=False
        with rcol:
            if st.session_state["pulse_ran"]:
                st.markdown('<div class="bubble system captionline">PULSECAST (MTD)</div>', unsafe_allow_html=True)
                st.markdown("> [ Outlook email screenshot / summary placeholder ]")

# ---------- ROW 10: Approve Pulse (only after Row9 right is visible) ----------
if st.session_state["pulse_ran"]:
    with row():
        lcol, rcol = st.columns([0.44,0.56], gap="small")
        with lcol:
            st.markdown('<div class="bubble">10) Do you approve the Pulse f’cast?</div>', unsafe_allow_html=True)
            c8,c9 = st.columns(2)
            if c8.button("YES ✅ (Pulse)", key="approve_pulse"):
                st.session_state["pulse_approved"]=True; st.session_state["pulse_df"]=spec_pulse_df.copy()
            if c9.button("NO, comment ✍️", key="reject_pulse"):
                st.session_state["pulse_approved"]=False
            if st.session_state["pulse_approved"] is False:
                st.text_input("Enter feedback / instruction for Pulsecast", key="pulse_comment")
                if st.button("Confirm Pulse note ✅", key="confirm_pulse_note"):
                    st.session_state["pulse_df"]=spec_pulse_df.copy(); st.session_state["pulse_confirmed"]=True
        with rcol:
            if (st.session_state["pulse_approved"] is True) or st.session_state["pulse_confirmed"]:
                st.dataframe(st.session_state["pulse_df"] if st.session_state["pulse_df"] is not None else spec_pulse_df, use_container_width=True)

# ---------- ROW 11: Additional Info (only after Row10 right is visible) ----------
can_add_info = (st.session_state["pulse_approved"] is True) or st.session_state["pulse_confirmed"]
if can_add_info:
    with row():
        lcol, rcol = st.columns([0.44,0.56], gap="small")
        with lcol:
            st.markdown('<div class="bubble">11) Add any additional info before Foresight?</div>', unsafe_allow_html=True)
            c10,c11 = st.columns(2)
            if c10.button("YES, add context 🧩", key="add_info_yes"):
                st.session_state["add_info_done"]=True
            if c11.button("NO", key="add_info_no"):
                st.session_state["add_info_done"]=True
            if st.session_state["add_info_done"]:
                st.session_state["add_info_text"]=st.text_area("Notes / context / events (optional)", height=100, key="addinfo")
        with rcol:
            if st.session_state["add_info_done"] and st.session_state["add_info_text"].strip():
                st.markdown(f"**User Context Added**: {st.session_state['add_info_text'].strip()}")

# ---------- ROW 12: Foresight (only after Row11 right is visible if any) ----------
if st.session_state["add_info_done"]:
    with row():
        lcol, rcol = st.columns([0.44,0.56], gap="small")
        with lcol:
            st.markdown('<div class="bubble">12) Trigger Foresight</div>', unsafe_allow_html=True)
            if st.button("Run Foresight 🤖➕", key="run_foresight"):
                b=np.array(base_values,float); f=np.array(festival_values,float)
                s=np.array(season_values,float) if st.session_state["season_df"] is None else np.array(st.session_state["season_df"]["Seasoncast Adjusted Forecast"].values,float)
                p=np.array(pulse_values,float)
                h=np.array(st.session_state["human_adj_df"]["Human-Adj Forecast"].values,float) if st.session_state["human_adj_df"] is not None else (b+f+s+p)/4.0
                A=np.vstack([b,f,s,p,h]).T; rng=np.random.default_rng(7); actuals=h*rng.normal(1.0,0.01,size=h.shape)
                w=learn_weights(A,actuals); 
                if w is None: w=np.array([0.25,0.15,0.25,0.25,0.10])
                foresight=(A*w).sum(axis=1)
                st.session_state["foresight_df"]=pd.DataFrame({"Date":dates,"Base":b.astype(int),"Festiv":f.astype(int),"Season":s.astype(int),"Pulse":p.astype(int),"Human":h.astype(int),"Final F’c":foresight.astype(int)})
                st.session_state["foresight_weights"]=dict(zip(["Base","Festival","Season","Pulse","Human"], np.round(w,3)))
                st.session_state["foresight_ran"]=True
        with rcol:
            if st.session_state["foresight_ran"] and st.session_state["foresight_df"] is not None:
                st.markdown('<div class="bubble system captionline">FORESIGHT THOUGHT & ACTIONS (LOG)</div>', unsafe_allow_html=True)
                st.markdown("Thought: adjust forecasts of Base, Festival, Season, Pulse, and Human (if any).  \nAction: learned weights from last 6 weeks + agent f’casts.")
                st.dataframe(st.session_state["foresight_df"], use_container_width=True)
                if st.session_state["foresight_weights"]:
                    weights_str="  ".join([f"{k}={v:.2f}" for k,v in st.session_state["foresight_weights"].items()])
                    st.markdown(f"**Weights used (learned)**: {weights_str}")

# ---------- ROW 13: Executive Summary (only after Row12 right is visible) ----------
if st.session_state["foresight_ran"]:
    with row():
        lcol, rcol = st.columns([0.44,0.56], gap="small")
        with lcol:
            st.markdown('<div class="bubble">13) Generate executive summary?</div>', unsafe_allow_html=True)
            c12,c13 = st.columns(2)
            if c12.button("YES, generate 📝", key="exec_yes"):
                w=st.session_state["foresight_weights"] or {"Base":0.25,"Festival":0.15,"Season":0.25,"Pulse":0.25,"Human":0.10}
                add=st.session_state["add_info_text"]
                summary=(f"**Why the forecast moved**: Blended weights leaned on Base/Season/Pulse, with weights {w}. "
                         f"Festival dampened peaks near holidays. Human inputs provided fine-grained guardrails.\n\n"
                         f"**Insights**: Stable last 6-week trend; mild pre-holiday softening and post-holiday reversion. "
                         f"Fraud signal flagged; volatility risk moderate.\n\n"
                         f"**Recommendations**: Guardrails ±3%, tighter anomaly windows around holidays, and consider daily AM Pulse ingestion.")
                if add and add.strip(): summary += f"\n\n**User Context Added**: {add.strip()}"
                st.session_state["exec_summary_text"]=summary; st.session_state["exec_summary_toggle"]=True
            if c13.button("NO", key="exec_no"):
                st.session_state["exec_summary_toggle"]=False
        with rcol:
            if st.session_state["exec_summary_toggle"] and st.session_state["exec_summary_text"]:
                st.markdown('<div class="bubble system captionline">EXECUTIVE SUMMARY</div>', unsafe_allow_html=True)
                st.markdown(st.session_state["exec_summary_text"])

# ---------- ROW 14: Email (only after Row13 right is visible) ----------
if st.session_state["exec_summary_toggle"]:
    with row():
        lcol, rcol = st.columns([0.44,0.56], gap="small")
        with lcol:
            st.markdown('<div class="bubble">14) Email the report?</div>', unsafe_allow_html=True)
            c14,c15 = st.columns(2)
            if c14.button("YES, email 📧", key="email_yes"):
                st.session_state["email_toggle"]=True
            if c15.button("NO", key="email_no"):
                st.session_state["email_toggle"]=False
            if st.session_state["email_toggle"]:
                st.session_state["email_to"]=st.text_input("To:", value=st.session_state["email_to"])
                st.session_state["email_subject"]=st.text_input("Subject:", value=st.session_state["email_subject"])
                if st.button("Send (simulate)", key="send_email"):
                    st.session_state["email_sent"]=True
        with rcol:
            if st.session_state["email_sent"]:
                st.success(f"Email sent to {st.session_state['email_to']!s} with subject “{st.session_state['email_subject']!s}”. (Simulated)")
