from typing import TypedDict , List  , Dict, Any 
from langgraph.graph import StateGraph , END       
# from langchain_community.chat_models import ChatOllama    
from langchain_ollama import ChatOllama 
from datetime import datetime, timedelta
import yfinance as yf 
import json  
import re
import pandas as pd 
  
 

# Define State     

class GraphState(TypedDict): 
    tickers: List[str]
    market_data: Dict[str, Dict[str, float]]
    thesis: dict[str, Any] 
    verifier_passed:bool  
    risk_view: dict[str, Any] 
    portfolio: Dict[str, float] 
    portfolio_checks: Dict[str, Any]   
    critique: dict[str, Any]
    explanation: str  
    
    
    # 2. Setup LLM  
    
# Thesis agent uses a bigger model for reasoning

thesis_llm = ChatOllama(
    model="athena-market-analyst:latest",
    temperature=0.2,
    format="json",
) 

# Risk agent uses a bigger model for reasoning
risk_llm = ChatOllama(
    model="mistral:7b",
    temperature=0.3,
    format="json",
) 


# Critics uses same big model but temperature for diversity 

critic_llm = ChatOllama(
    model="mistral:7b",
    temperature=0.5,
    format="json",
) 


# Explanation- narrative , no JSON contraint 
explanation_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.4,
)


# Helper 

def _safe_json(text: str) -> dict | None:
    
    """Best effort JSON extration from an LLM response."""
    
    if not text:
        return None 
    
    # strip code fences if present 
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    
    try:
        return json.loads(text)
    except Exception:
        
        # try to grab the first {...} block
        
        m= re.search(r"\{.*\}", text, re.DOTALL)
        
        if m: 
            try:
                return json.loads(m.group(0))
            except Exception:
                return None   
    return None


# Node 

# Node 1 -Market Data 

    # Market Data Node   
    
def market_data_node(state: GraphState) -> Dict[str, Any]:
    print("Running market_data_node....")
    data:Dict[str, Dict[str, float]] = {}
    
    for ticker in state["tickers"]:
        try:
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            
            # hist = stock.history(start= start, end= end)
            
            if hist.empty:
                continue
            
            start_price= float(hist["Close"].iloc[0])

            end_price= float(hist["Close"].iloc[-1])

            
            info = {}
            
            try:
                info = stock.info or {}
            except Exception:
                pass 
            
            data[ticker] = {
                "price": round(end_price, 4),
                "return_1m": round((end_price / start_price-1) * 100, 4),
                "pe_ratio": info.get("trailingPE"),
                "sector": info.get("sector"),
                
            }
            

            
        except Exception as e:
            print(f" X {ticker}: {e}")
            
            
    return {"market_data": data} 
            

# Node 2: Thesis Node  - LLM 

def thesis_node(state: GraphState)-> Dict[str, Any]:
    print("Running thesis_node(athena-market-analyst)....")
    
    prompt = f"""You are a senior equity research analyst at a long-only fund.
    
Market data (1-month window):
{json.dumps(state["market_data"], indent=2)}

For EACH ticker produce a recommendation.

Return ONLY this JSON shape:
{{
  "recommendations": [
    {{
      "ticker": "AAPL",
      "recommendation": "Buy" | "Hold" | "Sell",
      "confidence": <float 0..1>,
      "valuation_score": <float 0..1>,
      "reason": "<= 25 words"
    }}
  ]
}}

Rules:
- recommendation : {{"Buy","Hold","Sell"}}
- confidence : [0,1], higher = more conviction
- valuation_score: 0.8 = cheap, 0.5 = fair, 0.3 = expensive
- Do NOT invent data not present above.
- No markdown, no commentary outside JSON.
"""

    response = thesis_llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    
    parsed = _safe_json(raw)
    if parsed is None:
        
        # fall back to deterministic rules so the graph never dies
        
        print(" LLM thesis unparseable — falling back to rules")
        recs = []
        for ticker, info in state["market_data"].items():
            ret = info["return_1m"]
            if ret < -5:      rec = "Sell"
            elif ret <= 2:    rec = "Hold"
            else:             rec = "Buy"
            recs.append({
                "ticker": ticker,
                "recommendation": rec,
                "confidence": round(min(abs(ret) / 10, 1.0), 3),
                "valuation_score": 0.5,
                "reason": "deterministic fallback",
            })
        parsed = {"recommendations": recs}

    return {
        "thesis": {
            "raw": json.dumps(parsed),
            "parsed": parsed,
        }
    }

                                                                                                         
#Node 3 :  Verify Node

def verifier_node(state: GraphState) -> Dict[str, Any]:
    print("Running verifier_node...")
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
        if not all(k in r for k in ("ticker", "recommendation", "confidence")):
            return {"verifier_passed": False}
        if r["recommendation"] not in valid_labels:
            return {"verifier_passed": False}
        try:
            c = float(r["confidence"])
        except Exception:
            return {"verifier_passed": False}
        if not (0 <= c <= 1):
            return {"verifier_passed": False}
        
    return {"verifier_passed": True, "thesis": {"raw": raw, "parsed": parsed}}


# Node 4 - Risk Agent 

def risk_agent_node(state: GraphState) -> Dict[str, Any]:
    print("Running risk_agent_node (mistral:7b) ...")
    
    prompt = f"""You are a macro / risk analyst.
    
Tickers: {list(state["market_data"].keys())}
Thesis:  {json.dumps(state["thesis"]["parsed"], indent=2)}

Assess idiosyncratic and sector risk for each name.

Return ONLY:
{{
  "risk_flags": [
    {{
      "ticker": "AAPL",
      "risk_level": "low" | "medium" | "high",
      "risk_penalty": <float 0..0.4>,
      "reason": "<= 20 words"
    }}
  ],
  "market_regime": "risk-on" | "neutral" | "risk-off"
}}

Rules:
- risk_penalty = how much to shave off the portfolio weight (0 = none).
- Do not invent external news you weren't given.
- No markdown.
"""

    response = risk_llm.invoke(prompt)
    raw= response.content if hasattr(response, "content") else str(response)
    parsed = _safe_json(raw) or {"risk_flags": [], "market_regime": "neutral"}
    
    return {"risk_view": parsed}


# Node 5  Portfolio Node - risk aware 

def portfolio_node(state: GraphState) -> Dict[str, Any]:
    print("Running portfolio_node...")
    recs = state["thesis"]["parsed"]["recommendations"]
    
    recommendation_score_map = {"Buy": 3, "Hold": 2, "Sell": 1}
    
    # pull risk penalties
    risk_penalties = {}
    
    for f in state.get("risk_view", {}.get("risk_flags", [])):   
        try: 
            risk_penalties[f["ticker"]] = float(f.get("risk_flags", []))
        except Exception:
            pass

    scores: Dict[str, float] = {}
    
    for r in recs:
        rec_score = recommendation_score_map[r["recommendation"]]
        conf =  float(r["confidence"])
        val_score = float(r.get("valuation_score", 0.5))
        
        base = rec_score * 0.5 + conf * 0.3 + val_score * 0.2
        penalty = risk_penalties.get(r["ticker"], 0.0)
        
        scores[r["ticker"]] = max(base * (1 - penalty), 0.01)

        
        # score = (
        #     rec_score * 0.5 + conf * 0.3 + val_score * 0.2
        # )
        # scores[r["ticker"]] = max(score, 0.01)
        
    total= sum(scores.values())
    weights = {ticker: round(score / total * 100, 2) for ticker , score in scores.items()}
    
    return {"portfolio": weights}

# Node 6- Consistency verifier 

def consistency_verifier_node(state: GraphState) -> Dict[str, Any]:
    print("Running consistency_verifier_node...")
    recs = state["thesis"]["parsed"]["recommendations"]
    weights = state["portfolio"]
    
    issues: List[str] = []
    rec_map = {r["ticker"]: r for r in recs}
    
    total_weight = round(sum(weights.values()),2)
    if abs(total_weight - 100.0) > 1.0:
        issues.append(f"Portfolio weights sum to {total_weight}, not approximately 100.")
        
    for ticker ,weight in weights.items():
        rec = rec_map[ticker]
        label = rec["recommendation"]
        conf = float(rec["confidence"])
        
        if label == "Sell" and weight > 40:
            issues.append(f"{ticker} has a sell rating but weight {weight}% exceeds 40%.")
        if label == "Buy" and weight < 10:
            issues.append(f"{ticker} has a Buy rating but weight {weight}% is unexpectedly low.")
        if conf < 0.30 and weight > 35:
            issues.append(f"{ticker} has low confidence {conf} but very high weight {weight}%.")
            
            
                
    # Piar wise logic check 
    
    tickers = list(weights.keys())
    rank_map = {"Sell": 1, "Hold":2, "Buy": 3}
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            r1, r2 = rec_map[t1], rec_map[t2]
            w1, w2 = weights[t1], weights[t2]
            
            # rank_map = {"Sell": 1, "Hold":2, "Buy": 3}
            strength1= (rank_map[r1["recommendation"]], float(r1["confidence"]))
            strength2= (rank_map[r2["recommendation"]], float(r2["confidence"]))
            
            if strength1 > strength2 and w1 + 5 < w2:
                issues.append(f"{t1} appears stronger then {t2} but has materially lower weight ({w1}% vs {w2}%).")
                
    return {
        "portfolio_checks": {
            "passed": len(issues) == 0,
            "issues": issues
        }
    }
    
    
# Node 7- Critic 

def critic_node(state: GraphState) -> Dict[str, Any]:
    print("Running critic_node (mistral:7b) ...")

    prompt = f"""You are an independent portfolio critic. Be skeptical.

Thesis:     {json.dumps(state["thesis"]["parsed"], indent=2)}
Risk view:  {json.dumps(state.get("risk_view", {}), indent=2)}
Portfolio:  {json.dumps(state["portfolio"], indent=2)}
Checks:     {json.dumps(state.get("portfolio_checks", {}), indent=2)}


Return ONLY:
{{
  "overall_confidence": <float 0..1>,
  "strengths": ["..."],
  "concerns":  ["..."],
  "verdict":   "approve" | "revise"
}}

Rules:
- <= 3 items per list, each <= 20 words.
- Do not fabricate numbers.
"""

    response = critic_llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    parsed = _safe_json(raw) or {
        "overall_confidence": 0.5,
        "strengths": [], "concerns": [], "verdict": "approve",
    }
    return {"critique": parsed}


# Node 8 - Explanation

            
def explanation_node(state:GraphState) -> Dict[str, Any]:
    print("Running Explanation_node...")
    
    prompt = f"""
You are an equity research asistant.

Write a concise professional explanation of this portfolio result.

Market data:
{json.dumps(state["market_data"], indent=2)}

Thesis:
{json.dumps(state["thesis"]["parsed"], indent=2)}

Risk view:
{json.dumps(state.get("risk_view", {}), indent=2)}

Portfolio weights:
{json.dumps(state["portfolio"], indent=2)}

Consistency checks:
{json.dumps(state.get("portfolio_checks", {}), indent=2)}


Critique:
{json.dumps(state.get("critique", {}), indent=2)}

Instructions:
- Explain why each stock received its weight.
- Mention the 1-month return, recommendation, and confidence.
- Be clear that recommendation and confidence were generated using rule-based logic, not guessed by the model.
- Do not change any numbers.
- Keep it under 250 words.
"""

    response = explanation_llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return {"explanation": content}


# Conditional Logic

def check_verifier(state: GraphState) -> str:
    return "risk" if state["verifier_passed"] else END


# Build Graph

def build_graph():

    builder = StateGraph(GraphState)

    builder.add_node("market", market_data_node)
    builder.add_node("thesis", thesis_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("risk", risk_agent_node)
    builder.add_node("portfolio", portfolio_node)
    builder.add_node("consistency_verifier", consistency_verifier_node)
    builder.add_node("critic", critic_node)
    builder.add_node("explanation", explanation_node)


    builder.set_entry_point("market")

    builder.add_edge("market", "thesis")
    builder.add_edge("thesis", "verifier")

    builder.add_conditional_edges(
        "verifier",
        check_verifier,
        {
            "risk": "risk", END: END
        }
    )

    builder.add_edge("risk", "portfolio")
    builder.add_edge("portfolio", "consistency_verifier")
    builder.add_edge("consistency_verifier", "critic") 
    builder.add_edge("critic", "explanation") 
    builder.add_edge("explanation", END)

    return builder.compile()

graph = build_graph()


# Run 

if __name__ == "__main__":
    input_data = {"tickers": ["AAPL", "MSFT", "NVDA"]}
    result = graph.invoke(input_data)
    
    print("\nFINAL OUTPUT:\n")
    print(json.dumps(result, indent=2, default=str))
    
    
    print("\nLLM EXPLANATION:\n")
    print(result.get("explanation", "No explanation generated"))

       
            

 
        
    