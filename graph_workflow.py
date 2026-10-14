from typing import TypedDict , List  , Dict, Any 
from langgraph.graph import StateGraph , END       
# from langchain_community.chat_models import ChatOllama    
from langchain_ollama import ChatOllama 
from datetime import datetime, timedelta
import yfinance as yf 
import json  
  
 

# Define State     

class GraphState(TypedDict): 
    tickers: List[str]
    market_data: Dict[str, Dict[str, float]]
    thesis: dict[str, Any] 
    verifier_passed:bool   
    portfolio: Dict[str, float] 
    portfolio_checks: Dict[str, Any]   
    explanation: str  
    
    
    # 2. Setup LLM   
    
llm = ChatOllama(model="llama3.2:3b", temperature =0)

# Deterministic logic 

def classify_recommendation(return_1m):
    if return_1m < -5:
        return "Sell"
    elif return_1m <=2:
        return "Hold"
    else:
        return "Buy"

def compute_confidence(return_1m: float)-> float:
    return round(min(abs(return_1m) / 10, 1.0),3)

def valuation_score_from_pe(pe_ratio:float | None) -> float:
    if pe_ratio is None:
        return 0.5
    if pe_ratio < 20:
        return 0.8
    if pe_ratio < 35:
        return 0.6
    return 0.3

    
    #  Nodes 
    
    # Market Data Node   
    
def market_data_node(state: GraphState) -> Dict[str, Any]:
    print("Running market_data_node....")
    data:Dict[str, Dict[str, float]] = {}
    
    for ticker in state["tickers"]:
        stock = yf.Ticker(ticker)
        # end = datetime.today() - timedelta(days=1)    #
        # start = end - timedelta(days=30)              #
        
        hist = stock.history(period="1mo")
        
        # hist = stock.history(start= start, end= end)
        
        if hist.empty:
            continue
        start_price= float(hist["Close"].iloc[0])
        end_price= float(hist["Close"].iloc[-1])
        
        data[ticker] = {
            "price": round(end_price, 4),
            "return_1m": round((end_price / start_price-1) * 100, 4)
        }
            
    return {"market_data": data}

# Thesis Node (LLM) 

def thesis_node(state: GraphState)-> Dict[str, Any]:
    print("Running thesis_node....")
    recommendations = []
    
    for ticker , info in state["market_data"].items():
        ret = info["return_1m"]
        pe = info.get("pe_ratio")
        
        recommendations.append({
            "ticker": ticker,
            "recommendation": classify_recommendation(ret),
            "confidence": round(compute_confidence(ret), 3),
            "valuation_score": valuation_score_from_pe(pe),
        })
        
    thesis_obj = {"recommendations": recommendations}
        
    return {
        "thesis": {
            "raw": json.dumps(thesis_obj),
            "parsed": thesis_obj
        }
    }
                                                                                       
                    
# Verify Node

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
        if "ticker" not in r or "recommendation" not in r or "confidence" not in r:
            return {"verifier_passed": False}
        if r["recommendation"] not in valid_labels:
            return {"verifier_passed": False}
        if not (0 <= float(r["confidence"]) <= 1):
            return {"verifier_passed": False}
        
    return {"verifier_passed": True, "thesis": {"raw": raw, "parsed": parsed}}

# Portfolio Node 
def portfolio_node(state: GraphState) -> Dict[str, Any]:
    print("Running portfolio_node...")
    recs = state["thesis"]["parsed"]["recommendations"]
    
    recommendation_score_map = {"Buy": 3, "Hold": 2, "Sell": 1}
    scores: Dict[str, float] = {}
    
    for r in recs:
        rec_score = recommendation_score_map[r["recommendation"]]
        conf =  float(r["confidence"])
        val_score = float(r.get("valuation_score", 0.5))
        
        score = (
            rec_score * 0.5 + conf * 0.3 + val_score * 0.2
        )
        scores[r["ticker"]] = max(score, 0.01)
        
    total= sum(scores.values())
    weights = {ticker: round(score / total * 100, 2) for ticker , score in scores.items()}
    
    return {"portfolio": weights}

def consistency_verifier_node(state: GraphState) -> Dict[str, Any]:
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
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            r1, r2 = rec_map[t1], rec_map[t2]
            w1, w2 = weights[t1], weights[t2]
            
            rank_map = {"Sell": 1, "Hold":2, "Buy": 3}
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
            
def explanation_node(state:GraphState) -> Dict[str, Any]:
    print("Running Explanation_node...")
    
    prompt = f"""
You are an equity research asistant.

Write a concise professional expklanation of this portfolio result.

Market data:
{json.dumps(state["market_data"], indent=2)}

Deterministic thesis:
{json.dumps(state["thesis"]["parsed"], indent=2)}

Portfolio weights:
{json.dumps(state["portfolio"], indent=2)}

Consistency checks:
{json.dumps(state.get("portfolio_checks", {"passed": False, "issues": ["portfolio_checks missing"]}), indent=2)}

Instructions:
- Explain why each stock received its weight.
- Mention the 1-month return, recommendation, and confidence.
- Be clear that recommendation and confidence were generated using rule-based logic, not guessed by the model.
- Do not change any numbers.
- Keep it under 250 words.
"""

    response = llm.invoke(prompt)
    return {"explanation": response.content}


# Conditional Logic

def check_verifier(state: GraphState) -> str:
    return "portfolio" if state["verifier_passed"] else END


# Build Graph

def build_graph():

    builder = StateGraph(GraphState)

    builder.add_node("market", market_data_node)
    builder.add_node("thesis", thesis_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("portfolio", portfolio_node)
    builder.add_node("explanation", explanation_node)
    builder.add_node("consistency_verifier", consistency_verifier_node)

    builder.set_entry_point("market")

    builder.add_edge("market", "thesis")
    builder.add_edge("thesis", "verifier")

    builder.add_conditional_edges(
        "verifier",
        check_verifier,
        {
            "portfolio": "portfolio",
            END: END
        }
    )

    # builder.add_edge("portfolio", "explanation")
    builder.add_edge("portfolio", "consistency_verifier")
    builder.add_edge("consistency_verifier", "explanation") 
    builder.add_edge("explanation", END)

    return builder.compile()

graph = build_graph()


# Run 

if __name__ == "__main__":
    input_data = {
        "tickers": ["AAPL", "MSFT", "NVDA"]
    }
    
    result = graph.invoke(input_data)
    
    print("\nFINAL OUTPUT:\n")
    print(json.dumps(result, indent=2))
    
    
    print("\nLLM EXPLANATION:\n")
    print(result.get("explanation", "No explanation generated"))

       
            

 
        
    