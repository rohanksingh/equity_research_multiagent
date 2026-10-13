from typing import TypedDict, List
from langgraph.graph import StateGraph, END
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
import yfinance as yf


# --------------------------
# 1. Define State
# --------------------------

class GraphState(TypedDict):
    tickers: List[str]
    market_data: dict
    thesis: dict
    portfolio: dict
    verifier_passed: bool


# --------------------------
# 2. Setup LLM (Ollama)
# --------------------------

llm = ChatOllama(model="llama3.2:3b")


# --------------------------
# 3. Nodes
# --------------------------

# Market Data Node
def market_data_node(state: GraphState):
    data = {}

    for ticker in state["tickers"]:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")

        if not hist.empty:
            data[ticker] = {
                "price": float(hist["Close"].iloc[-1]),
                "return_1m": float((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100)
            }

    return {"market_data": data}


# Thesis Node (LLM)
def thesis_node(state: GraphState):
    prompt = f"""
You are an equity research assistant.

Using only this data:
{state['market_data']}

Return ONLY valid JSON in this format:
{{
  "recommendations": [
    {{
      "ticker": "AAPL",
      "recommendation": "Buy",
      "confidence": 0.65,
      "reason": "short explanation"
    }}
  ]
}}

Rules:
- recommendation must be one of: Buy, Hold, Sell
- confidence must be between 0 and 1
- no markdown
- no code
- no explanation outside JSON
"""

    response = llm.invoke(prompt)
    return {"thesis": {"raw": response.content}}


# Verifier Node
import json

def verifier_node(state: GraphState):
    raw = state["thesis"]["raw"]

    try:
        parsed = json.loads(raw)
    except Exception:
        return {"verifier_passed": False}

    if "recommendations" not in parsed:
        return {"verifier_passed": False}

    recs = parsed["recommendations"]
    if not isinstance(recs, list) or len(recs) == 0:
        return {"verifier_passed": False}

    valid_labels = {"Buy", "Hold", "Sell"}

    for r in recs:
        if "ticker" not in r or "recommendation" not in r or "confidence" not in r:
            return {"verifier_passed": False}
        if r["recommendation"] not in valid_labels:
            return {"verifier_passed": False}
        if not (0 <= float(r["confidence"]) <= 1):
            return {"verifier_passed": False}

    return {"verifier_passed": True, "thesis": {"raw": raw, "parsed": parsed}}


# Portfolio Node
def portfolio_node(state: GraphState):
    recs = state["thesis"]["parsed"]["recommendations"]

    score_map = {"Buy": 3, "Hold": 2, "Sell": 1}
    scores = {}

    for r in recs:
        score = score_map[r["recommendation"]] * float(r["confidence"])
        scores[r["ticker"]] = max(score, 0.01)

    total = sum(scores.values())
    weights = {ticker: round(score / total * 100, 2) for ticker, score in scores.items()}

    return {"portfolio": weights}


# --------------------------
# 4. Conditional Logic
# --------------------------

def check_verifier(state: GraphState):
    if state["verifier_passed"]:
        return "portfolio"
    else:
        return "thesis_retry"


def thesis_retry_node(state: GraphState):
    prompt = f"""
    Improve this thesis and fix issues:
    {state['thesis']}
    Ensure proper Buy/Hold/Sell recommendation.
    """

    response = llm.invoke(prompt)
    return {"thesis": {"raw": response.content}}


# --------------------------
# 5. Build Graph
# --------------------------

builder = StateGraph(GraphState)

builder.add_node("market", market_data_node)
builder.add_node("thesis", thesis_node)
builder.add_node("verifier", verifier_node)
builder.add_node("portfolio", portfolio_node)
builder.add_node("thesis_retry", thesis_retry_node)

builder.set_entry_point("market")

builder.add_edge("market", "thesis")
builder.add_edge("thesis", "verifier")

builder.add_conditional_edges(
    "verifier",
    check_verifier,
    {
        "portfolio": "portfolio",
        "thesis_retry": "thesis_retry"
    }
)

builder.add_edge("thesis_retry", "verifier")
builder.add_edge("portfolio", END)

graph = builder.compile()


# --------------------------
# 6. Run
# --------------------------

if __name__ == "__main__":
    input_data = {
        "tickers": ["AAPL", "MSFT", "NVDA"]
    }

    result = graph.invoke(input_data)

    print("\nFINAL OUTPUT:\n")
    print(result)