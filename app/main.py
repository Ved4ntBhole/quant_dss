# app/main.py  — QuantDSS India  (full-featured)
from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config_loader import get_policy, get_etf_universe
from src.data_loader import fetch_benchmark, fetch_single_ticker
from src.feature_engineering import FEATURE_COLS, compute_features_for_ticker
from src.model_trainer import load_model, predict_risk
from src.recommender import (
    _auto_provider,
    generate_recommendation,
    score_user_questionnaire,
    suggest_weight_for_asset,
)

_POLICY = get_policy()
CURRENCY = "₹"
BENCHMARK_NAME = "NIFTY 50"

# ── colour palette ────────────────────────────────────────────────────────────
RISK_COLOUR = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
RISK_BG     = {"Low": "#052e16", "Medium": "#451a03", "High": "#450a0a"}
RISK_EMOJI  = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

# ── load .env ─────────────────────────────────────────────────────────────────

def _load_env(path: str) -> bool:
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k and k not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
    return True


env_loaded = _load_env(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantDSS India",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* General */
[data-testid="stAppViewContainer"] { background: #0a0e1a; }
[data-testid="stSidebar"]          { background: #0d1117; border-right: 1px solid #1e2a3a; }
h1,h2,h3,h4                        { color: #e2e8f0; }
p, li, label, .stMarkdown           { color: #94a3b8; }

/* Risk badge */
.risk-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.85rem; font-weight: 700; letter-spacing: .05em;
}
/* Metric card */
.metric-card {
    background: #111827; border: 1px solid #1e2a3a; border-radius: 10px;
    padding: 16px 20px; text-align: center;
}
.metric-card .val { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; }
.metric-card .lbl { font-size: 0.75rem; color: #64748b; margin-top: 4px; }

/* ETF card */
.etf-card {
    background: #111827; border: 1px solid #1e2a3a; border-radius: 12px;
    padding: 18px 22px; margin-bottom: 12px;
}
.etf-ticker { font-size: 1.1rem; font-weight: 800; color: #60a5fa; }
.etf-name   { font-size: 0.82rem; color: #64748b; margin-bottom: 6px; }
.etf-pill   { display: inline-block; padding: 2px 10px; border-radius: 20px;
               font-size: 0.72rem; font-weight: 600; margin-right: 4px; }

/* Data source badge */
.source-badge {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
}

/* Insight row */
.insight-row {
    background: #0f172a; border-left: 3px solid #3b82f6;
    padding: 10px 16px; border-radius: 0 8px 8px 0; margin-bottom: 8px;
    color: #cbd5e1; font-size: 0.88rem;
}

/* Section header */
.section-hdr {
    font-size: 0.7rem; font-weight: 700; letter-spacing: .12em;
    color: #475569; text-transform: uppercase; margin: 24px 0 12px;
    padding-bottom: 6px; border-bottom: 1px solid #1e2a3a;
}

/* Warning / info banners */
.warn-box {
    background: #1c1000; border: 1px solid #92400e; color: #fbbf24;
    border-radius: 8px; padding: 10px 16px; font-size: 0.83rem; margin-top: 8px;
}
.info-box {
    background: #0c1a2e; border: 1px solid #1e40af; color: #93c5fd;
    border-radius: 8px; padding: 10px 16px; font-size: 0.83rem; margin-top: 8px;
}

/* Probability bar */
.prob-bar-wrap { background: #1e2a3a; border-radius: 6px; height: 8px; margin-top: 4px; }
.prob-bar      { border-radius: 6px; height: 8px; }

/* Report section */
.report-section {
    background: #111827; border: 1px solid #1e2a3a; border-radius: 10px;
    padding: 20px 24px; margin-bottom: 16px;
}

/* Divider */
.qdivider { border: none; border-top: 1px solid #1e2a3a; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)


# ── session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = dict(
        user_name="",
        user_risk=None,
        questionnaire_score=None,
        questionnaire_done=False,
        analysed_assets=[],        # list of dicts from analysis page
        current_page="questionnaire",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── helpers ───────────────────────────────────────────────────────────────────
def _risk_badge_html(risk: str) -> str:
    c = RISK_COLOUR.get(risk, "#94a3b8")
    bg = RISK_BG.get(risk, "#1e293b")
    e = RISK_EMOJI.get(risk, "⚪")
    return (f'<span class="risk-badge" '
            f'style="background:{bg};color:{c};border:1px solid {c};">'
            f'{e} {risk}</span>')


def _source_badge(provider: str) -> str:
    MAP = {
        "gemini":        ("🤖 Gemini AI", "#064e3b", "#34d399"),
        "alpha_vantage": ("📡 Alpha Vantage", "#0c1a2e", "#93c5fd"),
        "static":        ("⚙️ Static fallback", "#1c1000", "#fbbf24"),
    }
    label, bg, col = MAP.get(provider, ("⚙️ Unknown", "#1e293b", "#94a3b8"))
    return (f'<span class="source-badge" '
            f'style="background:{bg};color:{col};border:1px solid {col}40;">'
            f'{label}</span>')


def _pct(v, decimals=1):
    if v is None: return "—"
    return f"{v:+.{decimals}f}%"


def _fmt(v, decimals=2, prefix=""):
    if v is None: return "—"
    return f"{prefix}{v:.{decimals}f}"


# ── model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def _get_model():
    try:
        return (*load_model(), None)
    except FileNotFoundError as e:
        return None, None, str(e)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 QuantDSS India")
    st.caption("Quant-Driven Investment Decision Support System")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📝 Risk Questionnaire",
         "🔍 Live Asset Analysis",
         "💡 Recommendations",
         "📄 Report"],
        key="nav_page",
    )
    st.divider()

    # ── model status ──────────────────────────────────────────────────────────
    pipeline, le, model_err = _get_model()
    if model_err:
        st.error("❌ Model not found — run `python train.py` first.")
    else:
        st.success("✅ ML Model loaded")

    if env_loaded:
        st.caption("✅ .env loaded")

    # ── provider status ───────────────────────────────────────────────────────
    provider = _auto_provider()
    st.markdown("### 🔌 Provider Status")

    gemini_key   = bool(os.getenv("GEMINI_API_KEY", "").strip())
    alpha_key    = bool(os.getenv("ALPHA_VANTAGE_API_KEY", "").strip())
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    if provider == "gemini":
        st.success(f"🤖 Gemini AI active ({gemini_model})")
    elif provider == "alpha_vantage":
        st.info("📡 Alpha Vantage active")
    else:
        st.warning("⚠️ Static fallback — add API keys to .env")

    with st.expander("⚙️ Debug Info"):
        st.json({
            "provider"     : provider,
            "gemini_key"   : gemini_key,
            "gemini_model" : gemini_model,
            "alpha_key"    : alpha_key,
            "env_loaded"   : env_loaded,
            "benchmark"    : _POLICY.get("benchmark", "^NSEI"),
        })

    st.divider()

    # ── user profile summary ──────────────────────────────────────────────────
    if st.session_state.user_risk:
        st.markdown("### 👤 Current Profile")
        name = st.session_state.user_name or "Investor"
        risk = st.session_state.user_risk
        score = st.session_state.questionnaire_score
        st.markdown(f"**{name}**")
        st.markdown(_risk_badge_html(risk), unsafe_allow_html=True)
        if score is not None:
            st.caption(f"Questionnaire score: {score}/15")
        if st.button("🔄 Reset Profile", use_container_width=True):
            for k in ["user_name", "user_risk", "questionnaire_done",
                      "questionnaire_score", "analysed_assets"]:
                st.session_state[k] = "" if k == "user_name" else (
                    None if k in ["user_risk", "questionnaire_score"] else
                    False if k == "questionnaire_done" else [])
            st.rerun()

    st.divider()
    st.caption(f"Benchmark: {BENCHMARK_NAME}")
    st.caption("Price data: yfinance (NSE)")
    st.caption("AI text: Gemini API")
    st.caption(f"ML: RandomForest ({len(FEATURE_COLS)} features)")
    st.caption(f"Session: {datetime.now().strftime('%d %b %Y')}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — RISK QUESTIONNAIRE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📝 Risk Questionnaire":

    st.title("📝 Investor Risk Profiling")
    st.caption("Answer 5 questions to determine your risk tolerance. Takes under 2 minutes.")

    # ── score guide ───────────────────────────────────────────────────────────
    with st.expander("ℹ️ How scoring works"):
        bands = _POLICY.get("questionnaire_score_bands",
                             {"Low": [0, 4], "Medium": [5, 9], "High": [10, 15]})
        cols = st.columns(3)
        for i, (label, bounds) in enumerate(bands.items()):
            with cols[i]:
                st.markdown(_risk_badge_html(label), unsafe_allow_html=True)
                st.caption(f"Score {bounds[0]}–{bounds[1]} / 15")

    st.markdown('<hr class="qdivider">', unsafe_allow_html=True)

    with st.form("risk_form", clear_on_submit=False):
        name = st.text_input("Your name (optional)", value=st.session_state.user_name)

        c1, c2 = st.columns(2)
        with c1:
            horizon = st.selectbox(
                "1️⃣  Investment horizon",
                ["< 1 year", "1–3 years", "3–7 years", "7+ years"],
                help="How long can you keep your money invested?",
            )
            income = st.selectbox(
                "3️⃣  Income stability",
                ["Unstable", "Moderate", "Stable", "Very stable"],
                help="How reliable is your income source?",
            )
            experience = st.selectbox(
                "5️⃣  Investing experience",
                ["Beginner", "Some experience", "Experienced", "Professional"],
                help="How familiar are you with financial markets?",
            )
        with c2:
            loss = st.selectbox(
                "2️⃣  If your portfolio drops 20%…",
                ["Sell everything", "Sell some", "Hold", "Buy more"],
                help="What is your emotional reaction to losses?",
            )
            goal = st.selectbox(
                "4️⃣  Primary investment goal",
                ["Preserve capital", "Steady income", "Balanced growth", "Maximise gains"],
                help="What are you trying to achieve?",
            )

        submitted = st.form_submit_button("🎯 Calculate My Risk Profile",
                                          use_container_width=True, type="primary")

        if submitted:
            answers = dict(horizon=horizon, loss=loss, income=income,
                           goal=goal, experience=experience)
            risk = score_user_questionnaire(answers)

            # compute raw score for display
            from src.config_loader import get_questionnaire_scoring
            scoring = get_questionnaire_scoring()
            raw_score = sum(int(scoring.get(str(v), 0)) for v in answers.values())

            st.session_state.user_name        = name
            st.session_state.user_risk        = risk
            st.session_state.questionnaire_score = raw_score
            st.session_state.questionnaire_done  = True
            st.rerun()

    # ── result card ───────────────────────────────────────────────────────────
    if st.session_state.questionnaire_done and st.session_state.user_risk:
        risk = st.session_state.user_risk
        score = st.session_state.questionnaire_score
        name  = st.session_state.user_name or "Investor"

        st.markdown('<hr class="qdivider">', unsafe_allow_html=True)
        st.markdown("### Your Risk Profile")

        r1, r2, r3 = st.columns([2, 1, 1])
        with r1:
            st.markdown(f"## {RISK_EMOJI.get(risk,'')} {risk} Risk")
            st.markdown(f"Hello **{name}**, here is your personalised risk classification.")
        with r2:
            st.markdown(
                f'<div class="metric-card"><div class="val">{score}/15</div>'
                f'<div class="lbl">Questionnaire Score</div></div>',
                unsafe_allow_html=True,
            )
        with r3:
            bands = _POLICY.get("questionnaire_score_bands",
                                 {"Low": [0, 4], "Medium": [5, 9], "High": [10, 15]})
            low, high = bands.get(risk, [0, 15])
            pct = int((score - low) / max(high - low, 1) * 100)
            st.markdown(
                f'<div class="metric-card"><div class="val">{pct}%</div>'
                f'<div class="lbl">Within Band</div></div>',
                unsafe_allow_html=True,
            )

        desc = {
            "Low":    "You prefer capital preservation. You prioritise stability over high returns and are comfortable with lower growth to avoid significant losses.",
            "Medium": "You seek a balance between growth and stability. You can tolerate moderate fluctuations and are comfortable with a diversified portfolio.",
            "High":   "You are an aggressive investor. You seek maximum long-term returns and can withstand significant short-term volatility in pursuit of growth.",
        }
        c = RISK_COLOUR.get(risk, "#94a3b8")
        st.markdown(
            f'<div class="info-box" style="border-color:{c}40;color:{c}cc;">'
            f'{desc.get(risk, "")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Next step →** Go to **🔍 Live Asset Analysis** or **💡 Recommendations** in the sidebar.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — LIVE ASSET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Asset Analysis":

    st.title("🔍 Live Asset Analysis")

    if not st.session_state.user_risk:
        st.warning("⚠️ Please complete the **Risk Questionnaire** first.")
        st.stop()

    if model_err:
        st.error(f"❌ {model_err}")
        st.stop()

    user_risk = st.session_state.user_risk

    # ── header ────────────────────────────────────────────────────────────────
    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown(f"Analysing assets for investor profile: {_risk_badge_html(user_risk)}",
                    unsafe_allow_html=True)
    with hc2:
        period = st.selectbox("Data period", ["1y", "2y", "3y"], index=1,
                              label_visibility="collapsed")

    # ── ticker input ──────────────────────────────────────────────────────────
    ticker_input = st.text_input(
        "Enter NSE/BSE tickers (comma-separated)",
        value="RELIANCE.NS, INFY.NS, TCS.NS, HDFCBANK.NS, GOLDBEES.NS, NIFTYBEES.NS",
        help="Use .NS suffix for NSE (e.g. RELIANCE.NS) or .BO for BSE.",
    )

    col_btn, col_clear = st.columns([2, 1])
    with col_btn:
        analyse = st.button("🚀 Analyse Assets", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.analysed_assets = []
            st.rerun()

    if analyse:
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        if not tickers:
            st.warning("Please enter at least one ticker.")
            st.stop()

        results = []
        errors  = []

        with st.spinner(f"Fetching {len(tickers)} ticker(s) and computing features…"):
            try:
                benchmark = fetch_benchmark(period=period)
            except Exception as e:
                st.error(f"Could not fetch benchmark: {e}")
                st.stop()

            progress = st.progress(0)
            for idx, ticker in enumerate(tickers):
                progress.progress((idx + 1) / len(tickers), text=f"Processing {ticker}…")
                try:
                    df = fetch_single_ticker(ticker, period=period)
                    bm = benchmark.reindex(df.index).ffill()
                    feats = compute_features_for_ticker(df["Close"], bm)
                    if feats is None:
                        errors.append(f"{ticker}: insufficient data")
                        continue
                    pred = predict_risk(feats, pipeline, le)
                    results.append({
                        "ticker"                : ticker,
                        "risk_class"            : pred["predicted_class"],
                        "confidence"            : pred["confidence"],
                        "probabilities"         : pred["probabilities"],
                        **feats,
                    })
                except Exception as e:
                    errors.append(f"{ticker}: {e}")
            progress.empty()

        if errors:
            for err in errors:
                st.markdown(f'<div class="warn-box">⚠️ {err}</div>', unsafe_allow_html=True)

        st.session_state.analysed_assets = results

    # ── display results ───────────────────────────────────────────────────────
    assets = st.session_state.analysed_assets
    if not assets:
        st.markdown('<div class="info-box">Enter tickers above and click <b>Analyse Assets</b>.</div>',
                    unsafe_allow_html=True)
        st.stop()

    df_assets = pd.DataFrame(assets)

    # ── summary KPI row ───────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Portfolio Summary</p>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    n_low  = sum(1 for a in assets if a["risk_class"] == "Low")
    n_med  = sum(1 for a in assets if a["risk_class"] == "Medium")
    n_high = sum(1 for a in assets if a["risk_class"] == "High")
    n_match = sum(1 for a in assets if a["risk_class"] == user_risk)
    avg_vol = sum(a["annualised_volatility"] for a in assets) / len(assets)

    for col, val, lbl in zip(
        [k1, k2, k3, k4, k5],
        [f"{len(assets)}", f"🟢 {n_low}", f"🟡 {n_med}", f"🔴 {n_high}",
         f"{avg_vol:.1%}"],
        ["Assets Analysed", "Low Risk", "Medium Risk", "High Risk", "Avg Volatility"],
    ):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="val">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    # ── classification table ──────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Asset Classification Table</p>', unsafe_allow_html=True)

    display_cols = ["ticker", "risk_class", "confidence",
                    "annualised_return", "annualised_volatility",
                    "sharpe_ratio", "beta", "max_drawdown", "var_95", "downside_deviation"]

    df_display = df_assets[display_cols].copy()
    df_display["confidence"]           = df_display["confidence"].map("{:.1%}".format)
    df_display["annualised_return"]    = df_display["annualised_return"].map("{:+.1%}".format)
    df_display["annualised_volatility"]= df_display["annualised_volatility"].map("{:.1%}".format)
    df_display["sharpe_ratio"]         = df_display["sharpe_ratio"].map("{:.2f}".format)
    df_display["beta"]                 = df_display["beta"].map("{:.2f}".format)
    df_display["max_drawdown"]         = df_display["max_drawdown"].map("{:.1%}".format)
    df_display["var_95"]               = df_display["var_95"].map("{:.2%}".format)
    df_display["downside_deviation"]   = df_display["downside_deviation"].map("{:.1%}".format)

    df_display.columns = ["Ticker", "Risk Class", "Confidence",
                          "Ann. Return", "Ann. Volatility",
                          "Sharpe", "Beta", "Max Drawdown", "VaR 95%", "Downside Dev."]

    def _row_colour(row):
        c = RISK_COLOUR.get(row["Risk Class"], "#94a3b8") + "22"
        return [f"background-color:{c}"] * len(row)

    st.dataframe(
        df_display.style.apply(_row_colour, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # ── per-asset probability bars ────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Risk Probability Breakdown</p>', unsafe_allow_html=True)

    prob_cols = st.columns(min(len(assets), 3))
    for i, asset in enumerate(assets):
        with prob_cols[i % min(len(assets), 3)]:
            rk = asset["risk_class"]
            st.markdown(f"**{asset['ticker']}** {_risk_badge_html(rk)}",
                        unsafe_allow_html=True)
            probs = asset.get("probabilities", {})
            for label, colour in RISK_COLOUR.items():
                p = probs.get(label, 0)
                bar_w = max(1, int(p * 100))
                st.markdown(
                    f"<small style='color:{colour}'>{label}</small> "
                    f'<div class="prob-bar-wrap">'
                    f'<div class="prob-bar" style="width:{bar_w}%;background:{colour};"></div>'
                    f"</div>"
                    f"<small style='color:#64748b'>{p:.1%}</small>",
                    unsafe_allow_html=True,
                )
            st.markdown("---")

    # ── charts ────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Visualisations</p>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    # scatter: volatility vs return
    with ch1:
        fig_sc = px.scatter(
            df_assets,
            x="annualised_volatility",
            y="annualised_return",
            color="risk_class",
            text="ticker",
            size="confidence",
            color_discrete_map=RISK_COLOUR,
            title="Return vs Volatility",
            labels={
                "annualised_volatility": "Annualised Volatility",
                "annualised_return":     "Annualised Return",
                "risk_class":            "Risk Class",
            },
        )
        fig_sc.update_traces(textposition="top center", marker=dict(opacity=0.85))
        fig_sc.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font_color="#94a3b8", title_font_color="#e2e8f0",
            legend_title_font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e2a3a", tickformat=".0%"),
            yaxis=dict(gridcolor="#1e2a3a", tickformat=".0%"),
        )
        fig_sc.add_hline(y=0, line_dash="dot", line_color="#334155", annotation_text="0%")
        st.plotly_chart(fig_sc, use_container_width=True)

    # bar: Sharpe ratios
    with ch2:
        df_sorted = df_assets.sort_values("sharpe_ratio", ascending=True)
        fig_sh = go.Figure(go.Bar(
            x=df_sorted["sharpe_ratio"],
            y=df_sorted["ticker"],
            orientation="h",
            marker_color=[RISK_COLOUR.get(r, "#94a3b8") for r in df_sorted["risk_class"]],
            text=[f"{v:.2f}" for v in df_sorted["sharpe_ratio"]],
            textposition="outside",
        ))
        fig_sh.update_layout(
            title="Sharpe Ratio by Asset",
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font_color="#94a3b8", title_font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e2a3a"),
            yaxis=dict(gridcolor="#1e2a3a"),
            margin=dict(l=10, r=40, t=40, b=10),
        )
        fig_sh.add_vline(x=0, line_dash="dot", line_color="#334155")
        st.plotly_chart(fig_sh, use_container_width=True)

    # second row: Max drawdown + Beta
    ch3, ch4 = st.columns(2)
    with ch3:
        fig_dd = px.bar(
            df_assets.sort_values("max_drawdown", ascending=False),
            x="ticker", y="max_drawdown",
            color="risk_class", color_discrete_map=RISK_COLOUR,
            title="Max Drawdown",
            labels={"max_drawdown": "Max Drawdown", "ticker": ""},
        )
        fig_dd.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font_color="#94a3b8", title_font_color="#e2e8f0",
            yaxis=dict(gridcolor="#1e2a3a", tickformat=".0%"),
            xaxis=dict(gridcolor="#1e2a3a"),
            showlegend=False,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    with ch4:
        fig_beta = px.bar(
            df_assets.sort_values("beta"),
            x="ticker", y="beta",
            color="risk_class", color_discrete_map=RISK_COLOUR,
            title=f"Beta vs {BENCHMARK_NAME}",
            labels={"beta": "Beta", "ticker": ""},
        )
        fig_beta.add_hline(y=1, line_dash="dash", line_color="#475569",
                           annotation_text="β=1 (market)")
        fig_beta.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font_color="#94a3b8", title_font_color="#e2e8f0",
            yaxis=dict(gridcolor="#1e2a3a"),
            xaxis=dict(gridcolor="#1e2a3a"),
            showlegend=False,
        )
        st.plotly_chart(fig_beta, use_container_width=True)

    # ── profile match summary ─────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Profile Match</p>', unsafe_allow_html=True)
    matching  = [a for a in assets if a["risk_class"] == user_risk]
    outside   = [a for a in assets if a["risk_class"] != user_risk]

    pm1, pm2 = st.columns(2)
    with pm1:
        st.markdown(f"✅ **Matching your {user_risk} profile** ({len(matching)})")
        if matching:
            st.dataframe(
                pd.DataFrame([{"Ticker": a["ticker"],
                                "Return": f"{a['annualised_return']:+.1%}",
                                "Vol": f"{a['annualised_volatility']:.1%}",
                                "Sharpe": f"{a['sharpe_ratio']:.2f}",
                                "Confidence": f"{a['confidence']:.0%}"}
                               for a in matching]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("None of your assets match your risk profile.")
    with pm2:
        st.markdown(f"⚠️ **Outside your profile** ({len(outside)})")
        if outside:
            st.dataframe(
                pd.DataFrame([{"Ticker": a["ticker"],
                                "Classified As": a["risk_class"],
                                "Return": f"{a['annualised_return']:+.1%}",
                                "Vol": f"{a['annualised_volatility']:.1%}",
                                "Sharpe": f"{a['sharpe_ratio']:.2f}"}
                               for a in outside]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("All assets match your profile! 🎉")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Recommendations":

    st.title("💡 Investment Recommendations")

    if not st.session_state.user_risk:
        st.warning("⚠️ Please complete the **Risk Questionnaire** first.")
        st.stop()

    user_risk = st.session_state.user_risk
    user_name = st.session_state.user_name or "Investor"
    provider  = _auto_provider()

    # ── profile header ────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="report-section">'
        f'<h3 style="margin:0;color:#e2e8f0">INVESTOR PROFILE — {user_name}</h3>'
        f'<div style="margin-top:8px">{_risk_badge_html(user_risk)}'
        f' &nbsp; {_source_badge(provider)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── fetch recommendations ─────────────────────────────────────────────────
    with st.spinner("Fetching live NSE data and generating recommendations…"):
        try:
            rec = generate_recommendation(user_risk)
        except Exception as e:
            st.error(f"❌ Could not generate recommendations: {e}")
            st.stop()

    # ── warning banner ────────────────────────────────────────────────────────
    if rec.warning:
        st.markdown(f'<div class="warn-box">⚠️ {rec.warning}</div>',
                    unsafe_allow_html=True)

    if rec.ai_generated:
        st.markdown('<div class="info-box">🤖 AI-generated insights (Gemini)</div>',
                    unsafe_allow_html=True)

    # ── summary ───────────────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Portfolio Summary</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="report-section"><p style="color:#cbd5e1;margin:0;">{rec.summary}</p></div>',
        unsafe_allow_html=True,
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    if rec.etf_suggestions:
        sel = rec.etf_suggestions
        w_vol  = sum(e["annualised_volatility"] * e["suggested_weight"] for e in sel)
        w_ret  = sum(e["annualised_return"]     * e["suggested_weight"] for e in sel)
        w_beta = sum(e["beta"]                  * e["suggested_weight"] for e in sel)
        w_sh   = sum(e["sharpe_ratio"]           * e["suggested_weight"] for e in sel)

        k1, k2, k3, k4 = st.columns(4)
        for col, val, lbl in zip(
            [k1, k2, k3, k4],
            [f"{w_ret:+.1%}", f"{w_vol:.1%}", f"{w_beta:.2f}", f"{w_sh:.2f}"],
            ["Weighted Return", "Weighted Volatility", f"Weighted Beta vs {BENCHMARK_NAME}", "Weighted Sharpe"],
        ):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="val">{val}</div>'
                    f'<div class="lbl">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<hr class="qdivider">', unsafe_allow_html=True)

    # ── two-column layout: allocation chart + ETF cards ───────────────────────
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown('<p class="section-hdr">Sector Allocation</p>', unsafe_allow_html=True)
        if rec.sector_allocations:
            labels = list(rec.sector_allocations.keys())
            values = list(rec.sector_allocations.values())
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.55,
                marker=dict(colors=px.colors.qualitative.Set3),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            ))
            fig_pie.update_layout(
                paper_bgcolor="#111827",
                font_color="#94a3b8",
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                annotations=[dict(text=user_risk, x=0.5, y=0.5,
                                  font_size=16, font_color="#e2e8f0",
                                  showarrow=False)],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # allocation table
            alloc_data = [{"Sector": k, "Allocation": f"{v:.1f}%"}
                          for k, v in rec.sector_allocations.items()]
            st.dataframe(pd.DataFrame(alloc_data), use_container_width=True,
                         hide_index=True)

        # ── weight bar chart ──────────────────────────────────────────────────
        if rec.etf_suggestions:
            fig_wt = go.Figure(go.Bar(
                x=[e["suggested_weight"] * 100 for e in rec.etf_suggestions],
                y=[e["ticker"].replace(".NS","").replace(".BO","")
                   for e in rec.etf_suggestions],
                orientation="h",
                marker_color="#3b82f6",
                text=[f"{e['suggested_weight']*100:.0f}%"
                      for e in rec.etf_suggestions],
                textposition="outside",
            ))
            fig_wt.update_layout(
                title="Suggested Weights",
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font_color="#94a3b8", title_font_color="#e2e8f0",
                xaxis=dict(gridcolor="#1e2a3a", ticksuffix="%"),
                yaxis=dict(gridcolor="#1e2a3a"),
                margin=dict(l=10, r=40, t=40, b=10),
            )
            st.plotly_chart(fig_wt, use_container_width=True)

    with right_col:
        st.markdown('<p class="section-hdr">Recommended ETFs / Instruments</p>',
                    unsafe_allow_html=True)

        for etf in rec.etf_suggestions:
            ticker_disp = etf["ticker"].replace(".NS", "").replace(".BO", "")
            risk_bk     = etf.get("risk_bucket", "Medium")
            risk_c      = RISK_COLOUR.get(risk_bk, "#94a3b8")
            risk_bg     = RISK_BG.get(risk_bk, "#1e293b")
            cat         = etf.get("category", "")
            weight_pct  = f"{etf['suggested_weight']*100:.0f}%"

            # live price line
            price_str = ""
            if etf.get("live_price") is not None:
                price_str = f"{CURRENCY}{etf['live_price']:.2f}"
                if etf.get("day_change_pct") is not None:
                    dc = etf["day_change_pct"]
                    dc_col = "#22c55e" if dc >= 0 else "#ef4444"
                    dc_str = f"<span style='color:{dc_col}'>{dc:+.2%}</span>"
                    price_str += f" &nbsp; {dc_str}"

            vol_fmt  = f"{etf['annualised_volatility']:.1%}"
            ret_fmt  = f"{etf['annualised_return']:+.1%}"
            beta_fmt = f"{etf['beta']:.2f}"
            sh_fmt   = f"{etf['sharpe_ratio']:.2f}"
            dd_fmt   = f"{etf['max_drawdown']:.1%}"
            var_fmt  = f"{etf.get('var_95', 0):.2%}"

            st.markdown(
                f'<div class="etf-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div>'
                f'<span class="etf-ticker">{ticker_disp}</span>'
                f'<div class="etf-name">{etf["name"]}</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<span class="etf-pill" style="background:{risk_bg};color:{risk_c};border:1px solid {risk_c}40;">'
                f'{risk_bk}</span>'
                f'<span class="etf-pill" style="background:#0f172a;color:#60a5fa;border:1px solid #1e3a5f;">'
                f'weight {weight_pct}</span><br>'
                f'<span style="color:#94a3b8;font-size:0.78rem;">{cat}</span>'
                f'</div>'
                f'</div>'
                f'<hr style="border-color:#1e2a3a;margin:10px 0;">'
                f'<div style="display:flex;gap:16px;flex-wrap:wrap;font-size:0.82rem;">'
                f'<span>📈 <b style="color:#e2e8f0">{ret_fmt}</b> <span style="color:#475569">2y ret</span></span>'
                f'<span>〰️ <b style="color:#e2e8f0">{vol_fmt}</b> <span style="color:#475569">vol</span></span>'
                f'<span>β <b style="color:#e2e8f0">{beta_fmt}</b> <span style="color:#475569">vs NIFTY</span></span>'
                f'<span>⚡ <b style="color:#e2e8f0">{sh_fmt}</b> <span style="color:#475569">Sharpe</span></span>'
                f'<span>📉 <b style="color:#e2e8f0">{dd_fmt}</b> <span style="color:#475569">max DD</span></span>'
                f'<span>🎯 <b style="color:#e2e8f0">{var_fmt}</b> <span style="color:#475569">VaR 95%</span></span>'
                f'{f"<span>💰 {price_str}</span>" if price_str else ""}'
                f'</div>'
                f'<div style="margin-top:10px;color:#64748b;font-size:0.82rem;font-style:italic;">'
                f'{etf.get("reason","")}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── key insights ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Key Insights</p>', unsafe_allow_html=True)
    for insight in rec.key_insights:
        st.markdown(f'<div class="insight-row">💡 {insight}</div>',
                    unsafe_allow_html=True)

    # ── risk metrics deep-dive (radar / table) ────────────────────────────────
    if rec.etf_suggestions:
        st.markdown('<p class="section-hdr">ETF Risk Metrics Deep-Dive</p>',
                    unsafe_allow_html=True)
        metrics_df = pd.DataFrame([{
            "Ticker"    : e["ticker"].replace(".NS","").replace(".BO",""),
            "Category"  : e.get("category",""),
            "Return"    : f"{e['annualised_return']:+.1%}",
            "Volatility": f"{e['annualised_volatility']:.1%}",
            "Beta"      : f"{e['beta']:.2f}",
            "Sharpe"    : f"{e['sharpe_ratio']:.2f}",
            "Max DD"    : f"{e['max_drawdown']:.1%}",
            "VaR 95%"   : f"{e.get('var_95',0):.2%}",
            "Weight"    : f"{e['suggested_weight']*100:.0f}%",
        } for e in rec.etf_suggestions])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ── compared against analysed assets ─────────────────────────────────────
    analysed = st.session_state.analysed_assets
    if analysed:
        st.markdown('<p class="section-hdr">Your Analysed Assets vs Your Profile</p>',
                    unsafe_allow_html=True)
        matching = [a for a in analysed if a["risk_class"] == user_risk]
        outside  = [a for a in analysed if a["risk_class"] != user_risk]

        ma1, ma2 = st.columns(2)
        with ma1:
            st.markdown(f"✅ **Matching your {user_risk} profile** ({len(matching)})")
            if matching:
                st.dataframe(
                    pd.DataFrame([{
                        "Ticker"  : a["ticker"],
                        "Return"  : f"{a['annualised_return']:+.1%}",
                        "Vol"     : f"{a['annualised_volatility']:.1%}",
                        "Sharpe"  : f"{a['sharpe_ratio']:.2f}",
                        "Beta"    : f"{a['beta']:.2f}",
                        "ML Conf.": f"{a['confidence']:.0%}",
                    } for a in matching]),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("None of your assets match your risk profile.")
        with ma2:
            st.markdown(f"⚠️ **Outside your profile** ({len(outside)})")
            if outside:
                st.dataframe(
                    pd.DataFrame([{
                        "Ticker"       : a["ticker"],
                        "Classified As": a["risk_class"],
                        "Return"       : f"{a['annualised_return']:+.1%}",
                        "Vol"          : f"{a['annualised_volatility']:.1%}",
                        "Sharpe"       : f"{a['sharpe_ratio']:.2f}",
                    } for a in outside]),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("All your assets match your profile! 🎉")

    # ── disclaimer ────────────────────────────────────────────────────────────
    st.markdown('<hr class="qdivider">', unsafe_allow_html=True)
    st.caption(
        "⚠️ **Disclaimer:** This is a research tool, not SEBI-registered investment advice. "
        "Past performance is not indicative of future results. "
        "Mutual fund investments are subject to market risks. Please read all scheme-related "
        "documents carefully before investing."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — REPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Report":

    st.title("📄 Investment Report")

    if not st.session_state.user_risk:
        st.warning("⚠️ Please complete the **Risk Questionnaire** first.")
        st.stop()

    user_risk = st.session_state.user_risk
    user_name = st.session_state.user_name or "Investor"
    provider  = _auto_provider()
    generated = datetime.now().strftime("%d %B %Y, %H:%M")

    # ── fetch recommendations ─────────────────────────────────────────────────
    with st.spinner("Preparing report…"):
        try:
            rec = generate_recommendation(user_risk)
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()

    # ── report header ─────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="report-section">'
        f'<h2 style="margin:0;color:#e2e8f0">QuantDSS India — Investment Report</h2>'
        f'<p style="margin:4px 0 0;color:#64748b">Generated: {generated}</p>'
        f'<hr style="border-color:#1e2a3a;margin:12px 0;">'
        f'<b style="color:#94a3b8">Investor:</b> '
        f'<span style="color:#e2e8f0">{user_name}</span> &nbsp;|&nbsp; '
        f'<b style="color:#94a3b8">Risk Profile:</b> '
        f'{_risk_badge_html(user_risk)} &nbsp;|&nbsp; '
        f'<b style="color:#94a3b8">Data Source:</b> {_source_badge(provider)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if rec.warning:
        st.markdown(f'<div class="warn-box">⚠️ {rec.warning}</div>',
                    unsafe_allow_html=True)

    # ── section 1: summary ────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">1 · Portfolio Summary</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="report-section"><p style="color:#cbd5e1;margin:0;">{rec.summary}</p></div>',
        unsafe_allow_html=True,
    )

    # ── section 2: KPIs ───────────────────────────────────────────────────────
    if rec.etf_suggestions:
        sel    = rec.etf_suggestions
        w_vol  = sum(e["annualised_volatility"] * e["suggested_weight"] for e in sel)
        w_ret  = sum(e["annualised_return"]     * e["suggested_weight"] for e in sel)
        w_beta = sum(e["beta"]                  * e["suggested_weight"] for e in sel)
        w_sh   = sum(e["sharpe_ratio"]           * e["suggested_weight"] for e in sel)

        st.markdown('<p class="section-hdr">2 · Key Metrics</p>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        for col, val, lbl in zip(
            [k1, k2, k3, k4],
            [f"{w_ret:+.1%}", f"{w_vol:.1%}", f"{w_beta:.2f}", f"{w_sh:.2f}"],
            ["Weighted Return", "Weighted Volatility", "Portfolio Beta", "Portfolio Sharpe"],
        ):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="val">{val}</div>'
                    f'<div class="lbl">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

    # ── section 3: sector allocation ──────────────────────────────────────────
    st.markdown('<p class="section-hdr">3 · Sector Allocation</p>', unsafe_allow_html=True)
    s1, s2 = st.columns([1, 2])
    with s1:
        if rec.sector_allocations:
            st.dataframe(
                pd.DataFrame([{"Sector": k, "Weight": f"{v:.1f}%"}
                               for k, v in rec.sector_allocations.items()]),
                use_container_width=True, hide_index=True,
            )
    with s2:
        if rec.sector_allocations:
            fig_donut = go.Figure(go.Pie(
                labels=list(rec.sector_allocations.keys()),
                values=list(rec.sector_allocations.values()),
                hole=0.5,
                marker=dict(colors=px.colors.qualitative.Pastel),
                textinfo="label+percent",
            ))
            fig_donut.update_layout(
                paper_bgcolor="#111827", font_color="#94a3b8",
                showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                annotations=[dict(text=user_risk, x=0.5, y=0.5,
                                  font_size=14, font_color="#e2e8f0",
                                  showarrow=False)],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    # ── section 4: recommended instruments ────────────────────────────────────
    st.markdown('<p class="section-hdr">4 · Recommended Instruments</p>',
                unsafe_allow_html=True)
    if rec.etf_suggestions:
        st.dataframe(
            pd.DataFrame([{
                "Ticker"    : e["ticker"].replace(".NS","").replace(".BO",""),
                "Name"      : e["name"],
                "Category"  : e.get("category",""),
                "Return"    : f"{e['annualised_return']:+.1%}",
                "Volatility": f"{e['annualised_volatility']:.1%}",
                "Beta"      : f"{e['beta']:.2f}",
                "Sharpe"    : f"{e['sharpe_ratio']:.2f}",
                "Max DD"    : f"{e['max_drawdown']:.1%}",
                "VaR 95%"   : f"{e.get('var_95',0):.2%}",
                "Weight"    : f"{e['suggested_weight']*100:.0f}%",
                "Rationale" : e.get("reason", ""),
            } for e in rec.etf_suggestions]),
            use_container_width=True, hide_index=True,
        )

    # ── section 5: key insights ───────────────────────────────────────────────
    st.markdown('<p class="section-hdr">5 · Key Insights</p>', unsafe_allow_html=True)
    for i, insight in enumerate(rec.key_insights, 1):
        st.markdown(f'<div class="insight-row">{i}. {insight}</div>',
                    unsafe_allow_html=True)

    # ── section 6: analysed assets ────────────────────────────────────────────
    analysed = st.session_state.analysed_assets
    if analysed:
        st.markdown('<p class="section-hdr">6 · Your Analysed Assets</p>',
                    unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([{
                "Ticker"     : a["ticker"],
                "Risk Class" : a["risk_class"],
                "Profile Match": "✅" if a["risk_class"] == user_risk else "⚠️",
                "Return"     : f"{a['annualised_return']:+.1%}",
                "Volatility" : f"{a['annualised_volatility']:.1%}",
                "Sharpe"     : f"{a['sharpe_ratio']:.2f}",
                "Beta"       : f"{a['beta']:.2f}",
                "Max DD"     : f"{a['max_drawdown']:.1%}",
                "ML Conf."   : f"{a['confidence']:.0%}",
            } for a in analysed]),
            use_container_width=True, hide_index=True,
        )

    # ── download buttons ──────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Downloads</p>', unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)

    # JSON report
    report_json = {
        "generated"      : generated,
        "investor"       : user_name,
        "risk_profile"   : user_risk,
        "data_source"    : provider,
        "ai_generated"   : rec.ai_generated,
        "summary"        : rec.summary,
        "kpis"           : {
            "weighted_return"    : f"{sum(e['annualised_return']*e['suggested_weight'] for e in rec.etf_suggestions):+.2%}" if rec.etf_suggestions else "—",
            "weighted_volatility": f"{sum(e['annualised_volatility']*e['suggested_weight'] for e in rec.etf_suggestions):.2%}" if rec.etf_suggestions else "—",
            "weighted_beta"      : f"{sum(e['beta']*e['suggested_weight'] for e in rec.etf_suggestions):.2f}" if rec.etf_suggestions else "—",
            "weighted_sharpe"    : f"{sum(e['sharpe_ratio']*e['suggested_weight'] for e in rec.etf_suggestions):.2f}" if rec.etf_suggestions else "—",
        },
        "sector_allocations": rec.sector_allocations,
        "etf_suggestions"   : [
            {k: v for k, v in e.items() if k != "probabilities"}
            for e in rec.etf_suggestions
        ],
        "key_insights"      : rec.key_insights,
        "analysed_assets"   : [
            {k: v for k, v in a.items() if k != "probabilities"}
            for a in analysed
        ],
    }

    with dl1:
        st.download_button(
            "📥 Download JSON Report",
            data=json.dumps(report_json, indent=2, default=str),
            file_name=f"quantdss_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )

    # CSV — ETF suggestions
    if rec.etf_suggestions:
        csv_df = pd.DataFrame([
            {k: v for k, v in e.items() if k not in ("probabilities",)}
            for e in rec.etf_suggestions
        ])
        with dl2:
            st.download_button(
                "📥 Download ETFs CSV",
                data=csv_df.to_csv(index=False),
                file_name=f"quantdss_etfs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # CSV — analysed assets
    if analysed:
        csv_assets = pd.DataFrame([
            {k: v for k, v in a.items() if k != "probabilities"}
            for a in analysed
        ])
        with dl3:
            st.download_button(
                "📥 Download Assets CSV",
                data=csv_assets.to_csv(index=False),
                file_name=f"quantdss_assets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── PDF report ────────────────────────────────────────────────────────────
    try:
        from fpdf import FPDF

        def _build_pdf(report: dict, rec_obj, user: str, risk: str) -> bytes:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # title
            pdf.set_font("Helvetica", "B", 20)
            pdf.set_text_color(30, 64, 175)
            pdf.cell(0, 12, "QuantDSS India — Investment Report", ln=True)

            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 6, f"Generated: {report['generated']}", ln=True)
            pdf.cell(0, 6, f"Investor: {user}   |   Risk Profile: {risk}", ln=True)
            pdf.cell(0, 6, f"Data source: {report['data_source']}", ln=True)
            pdf.ln(4)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(6)

            def _section(title):
                pdf.set_font("Helvetica", "B", 12)
                pdf.set_text_color(30, 64, 175)
                pdf.cell(0, 8, title, ln=True)
                pdf.set_text_color(30, 41, 59)

            # summary
            _section("1. Portfolio Summary")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(71, 85, 105)
            pdf.multi_cell(0, 6, rec_obj.summary)
            pdf.ln(4)

            # KPIs
            _section("2. Key Metrics")
            for k, v in report.get("kpis", {}).items():
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(71, 85, 105)
                pdf.cell(0, 6, f"  {k.replace('_',' ').title()}: {v}", ln=True)
            pdf.ln(4)

            # sector allocations
            _section("3. Sector Allocation")
            for sector, pct in rec_obj.sector_allocations.items():
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(71, 85, 105)
                pdf.cell(0, 6, f"  {sector}: {pct:.1f}%", ln=True)
            pdf.ln(4)

            # ETF suggestions
            _section("4. Recommended Instruments")
            for etf in rec_obj.etf_suggestions:
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(30, 41, 59)
                ticker = etf["ticker"].replace(".NS","").replace(".BO","")
                pdf.cell(0, 7, f"  {ticker} — {etf['name']}  (weight {etf['suggested_weight']*100:.0f}%)", ln=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(100, 116, 139)
                pdf.cell(0, 5,
                    f"    Return {etf['annualised_return']:+.1%} | Vol {etf['annualised_volatility']:.1%} | "
                    f"Beta {etf['beta']:.2f} | Sharpe {etf['sharpe_ratio']:.2f} | DD {etf['max_drawdown']:.1%}",
                    ln=True,
                )
                if etf.get("reason"):
                    pdf.multi_cell(0, 5, f"    {etf['reason']}")
                pdf.ln(2)
            pdf.ln(2)

            # insights
            _section("5. Key Insights")
            for i, ins in enumerate(rec_obj.key_insights, 1):
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(71, 85, 105)
                pdf.multi_cell(0, 6, f"  {i}. {ins}")
            pdf.ln(4)

            # analysed assets
            if report.get("analysed_assets"):
                _section("6. Analysed Assets")
                for a in report["analysed_assets"]:
                    pdf.set_font("Helvetica", "", 9)
                    pdf.set_text_color(71, 85, 105)
                    pdf.cell(0, 5,
                        f"  {a['ticker']} — {a['risk_class']} risk (conf {a['confidence']:.0%}) | "
                        f"Return {a['annualised_return']:+.1%} | Vol {a['annualised_volatility']:.1%} | "
                        f"Sharpe {a['sharpe_ratio']:.2f}",
                        ln=True,
                    )
                pdf.ln(4)

            # disclaimer
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(148, 163, 184)
            pdf.multi_cell(0, 5,
                "Disclaimer: This report is generated by QuantDSS, a research and educational tool. "
                "It does not constitute SEBI-registered investment advice. Mutual fund investments are "
                "subject to market risks. Past performance is not indicative of future results."
            )

            return pdf.output(dest="S").encode("latin-1")

        pdf_bytes = _build_pdf(report_json, rec, user_name, user_risk)
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"quantdss_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=False,
        )

    except ImportError:
        st.caption("Install `fpdf2` to enable PDF download.")
    except Exception as e:
        st.caption(f"PDF generation skipped: {e}")

    # ── disclaimer ────────────────────────────────────────────────────────────
    st.markdown('<hr class="qdivider">', unsafe_allow_html=True)
    st.caption(
        "⚠️ **Disclaimer:** This report is for research and educational purposes only. "
        "It does not constitute SEBI-registered investment advice. "
        "Mutual fund investments are subject to market risks. "
        "Please read all scheme-related documents carefully before investing."
    )