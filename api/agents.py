from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage
from tools.yahooFinanceNewsTool import stock_news
from api.tools.stockPriceAnaliserTool import rsi_predictor
from api.tools.PDPTool import pdp_news_scraper 
import json

# "ARDYZ.IS", "YEOTK.IS", "EGPRO.IS", "VESTL.IS", "NATEN.IS","SELEC.IS", "HRKET.IS", "MIATK.IS", "CWENE.IS", "ISDMR.IS", "TUKAS.IS" create a strategy 

BASE_URL = "http://ollama_server:11434"

model = ChatOllama(model="llama3.1:latest", base_url=BASE_URL, verbose=True)

stock_price_analiser_agent = create_react_agent(
    model=model,
    tools=[rsi_predictor],
    name="stock_price_expert",
    prompt="""
    You are a technical analysis expert.
    You should analysis data for given stock.
    Use tool to evaluate:
    - signal, ema34, ema89, price_above_ema34
    - rsi_fibo_divergence, channel_position

    Output a short summary explaining the technical setup and sentiment (bullish, bearish, neutral). Keep it concise and data-driven.
    """
)

stock_news_agent = create_react_agent(
    model=model,
    tools=[stock_news],
    name="stock_news_expert",
    prompt="""
    You are a stock news risk analyst.

    Analyze recent news and assign a **Risk Score (1–10)** with 2–3 sentence reasoning.

    === Risk Assessment ===
    Risk Score: <score>/10
    Reasoning: <short explanation>
    =======================
    """
)


pdp_agent = create_react_agent(
    model=model,
    tools=[pdp_news_scraper],
    name="pdp_expert",
    prompt="""
    You analyze PDP (KAP) filings.

    Provide:
    - Summary of the latest disclosure
    - Market impact (Positive/Negative/Neutral)
    - 2–3 sentence explanation

    === PDP Analysis ===
    Summary: <brief summary>
    Market Impact: <Positive | Negative | Neutral>
    Reasoning: <short explanation>
    ======================
    """
)

workflow = create_supervisor(
    [stock_price_analiser_agent, stock_news_agent, pdp_agent],
    model=model,
    prompt="""
    You are a supervisor agent overseeing a team of financial analysis experts:

    1. **Stock Price Analyst**: Evaluates RSI, EMA34/EMA89, signal strength, price divergence, and momentum.
    2. **News Risk Analyst**: Scores financial news (1–10) with risk interpretation.
    3. **PDP Analyst**: Evaluates disclosures from PDP (KAP) for market impact.

    You are responsible for reviewing their outputs and producing a comprehensive investment report for each stock ticker provided.

    For each ticker:
    - Combine all insights into a well-structured financial report.
    - Your goal is to **maximize potential return with calculated risk**.
    - Ensure **price data is verified from two independent calls** to reduce error.
    - Present **all monetary values in Turkish Lira (₺)**.

    The report must include:

    =======================
    📊 Ticker: <SYMBOL>

    🔍 Technical Summary:
    - Signal: <bullish/bearish/neutral>
    - EMA34: ₺<value>
    - EMA89: ₺<value>
    - Price Above EMA34: <true/false>
    - RSI/Fibonacci Divergence: <yes/no>
    - Channel Position: <top/middle/bottom>

    📅 Next Day Outlook:
    - Forecasted Price Range: ₺<low> - ₺<high>
    - Decision: <Buy | Hold | Sell>
    - Reasoning: <short 2–3 sentence justification based on short-term indicators>

    📆 Strategy Recommendations:
    - Short-Term (1–7d): <Buy | Hold | Sell> → <reason>
    - Mid-Term (1w–1mo): <Buy | Hold | Sell> → <reason>
    - Long-Term (1–6mo): <Buy | Hold | Sell> → <reason>

    ⚠️ Risk Analysis:
    === Risk Assessment ===
    Risk Score: <x>/10
    Reasoning: <short explanation>
    =======================

    🧾 PDP Filing Summary:
    === PDP Analysis ===
    Summary: <brief summary>
    Market Impact: <Positive | Negative | Neutral>
    Reasoning: <brief impact analysis>
    ======================

    ✅ Final Action Plan:
    Recommendation: <e.g., Buy on pullback, Aggressive Hold, Avoid Short-Term>
    Suggested Strategy: <e.g., RSI breakout entry, EMA trend-following, Mean-reversion>

    =======================

    You must make bold but well-reasoned decisions. Use all data to guide the investor with confidence.
    """,
    output_mode="full_history"
)

finance_app = workflow.compile()

def read_root():
    return {"message": "🎉 FastAPI is working!"}

def ask(question: str):
    response = finance_app.invoke({"messages": [HumanMessage(question)]})
    output = response.get("output", str(response))
    return {"output": output}