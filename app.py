import json
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── import the compiled graph from your existing module ──
from graph_workflow import graph


# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Equity Research Multi-Agent",
    page_icon="",
    layout="wide",
)

st.title("Equity Research Multi-Agent")
st.caption(
    "LangGraph + Ollama + yfinance · deterministic thesis + "
    "rule-based verifier + LLM explanation"
)


# ═══════════════════════════════════════════════════════════
# SIDEBAR — inputs
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Configuration")

    raw = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, NVDA",
        help="e.g. AAPL, MSFT, NVDA, TSLA",
    )

    run_btn = st.button("▶ Run workflow", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown(
        """
        1. `market` — yfinance 1-month history
        2. `thesis` — rule-based Buy/Hold/Sell
        3. `verifier` — schema + label check
        4. `portfolio` — weighted allocation
        5. `consistency_verifier` — cross-checks
        6. `explanation` — LLM narrative
        """
    )


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def parse_tickers(text: str) -> list[str]:
    return [t.strip().upper() for t in text.split(",") if t.strip()]


def market_df(market_data: dict) -> pd.DataFrame:
    if not market_data:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(market_data, orient="index").reset_index()
    df.rename(columns={"index": "ticker"}, inplace=True)
    # enforce column order, add missing as NaN
    for col in ["ticker", "price", "return_1m", "pe_ratio", "sector"]:
        if col not in df.columns:
            df[col] = None
    return df[["ticker", "price", "return_1m", "pe_ratio", "sector"]]


def thesis_df(thesis_parsed: dict) -> pd.DataFrame:
    recs = thesis_parsed.get("recommendations", []) if thesis_parsed else []
    if not recs:
        return pd.DataFrame()
    return pd.DataFrame(recs)


def portfolio_df(portfolio: dict) -> pd.DataFrame:
    if not portfolio:
        return pd.DataFrame()
    df = pd.DataFrame(
        {"ticker": list(portfolio.keys()), "weight_%": list(portfolio.values())}
    )
    return df.sort_values("weight_%", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════

if run_btn:
    tickers = parse_tickers(raw)
    if not tickers:
        st.error("Enter at least one ticker.")
        st.stop()

    st.info(f"Running workflow for: **{', '.join(tickers)}**")

    progress = st.progress(0, text="Starting …")

    # Show each node as it prints by capturing stdout
    import io, contextlib

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = graph.invoke({"tickers": tickers})
        progress.progress(100, text="Done.")
    except Exception as e:
        progress.empty()
        st.exception(e)
        st.stop()

    log = buf.getvalue()

    # ── persist across reruns ──
    st.session_state["result"] = result
    st.session_state["log"] = log


# ═══════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════

result = st.session_state.get("result")
log = st.session_state.get("log", "")

if not result:
    st.info("Configure tickers in the sidebar and click **Run workflow**.")
    st.stop()


# ── run log (collapsible) ──
with st.expander(" Run log", expanded=False):
    st.code(log or "(empty)", language="text")


# ── top KPI row ──
recs = result.get("thesis", {}).get("parsed", {}).get("recommendations", [])
checks = result.get("portfolio_checks", {})
passed = checks.get("passed", False)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers", len(result.get("market_data", {})))
c2.metric("Recommendations", len(recs))
c3.metric(
    "Consistency",
    "PASS ✅" if passed else "FAIL ❌",
    delta=None if passed else f"{len(checks.get('issues', []))} issue(s)",
    delta_color="inverse",
)
c4.metric(
    "Weights sum",
    f"{round(sum(result.get('portfolio', {}).values()), 2)} %",
)

st.divider()


# ── tabs ──
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [" Market Data", " Thesis", " Portfolio", " Checks", " Explanation"]
)

with tab1:
    st.subheader("Market data (1-month window)")
    df = market_df(result.get("market_data", {}))
    if df.empty:
        st.warning("No market data returned.")
    else:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "price": st.column_config.NumberColumn(format="%.2f"),
                "return_1m": st.column_config.NumberColumn(
                    "return_1m (%)", format="%.2f%%"
                ),
                "pe_ratio": st.column_config.NumberColumn(format="%.2f"),
            },
        )

with tab2:
    st.subheader("Deterministic thesis")
    st.caption(
        "Generated by rule-based logic in `thesis_node` "
        "(return thresholds + PE valuation score)."
    )
    df = thesis_df(result.get("thesis", {}).get("parsed", {}))
    if df.empty:
        st.warning("No recommendations produced.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Portfolio weights")
    df = portfolio_df(result.get("portfolio", {}))
    if df.empty:
        st.warning("No portfolio produced.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("ticker")["weight_%"])

with tab4:
    st.subheader("Consistency checks")
    if passed:
        st.success("All checks passed.")
    else:
        st.error(f"{len(checks.get('issues', []))} issue(s) found:")
        for issue in checks.get("issues", []):
            st.write(f"- {issue}")

    st.subheader("Verifier gate")
    st.write(f"`verifier_passed` = **{result.get('verifier_passed')}**")

with tab5:
    st.subheader("LLM explanation")
    st.write(result.get("explanation", "_No explanation generated._"))


# ── raw JSON ──
with st.expander("Raw state (JSON)"):
    st.json(result, expanded=False)