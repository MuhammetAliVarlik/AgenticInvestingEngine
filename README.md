# 🤖 AgenticInvestingEngine

AI-Powered Financial Analysis & Investment Strategy API

This project is a containerized microservice-based system built with FastAPI, LangChain, and Docker, leveraging **LLM-powered agents** to analyze **BIST100 stocks** using technical indicators, financial news, and PDP (Public Disclosure Platform) filings — generating intelligent investment strategies.

---

## 🚀 Features

- 🧠 Uses **Llama 3.1** via ChatOllama for advanced language understanding  
- 📊 **Technical analysis tools** (RSI, EMA34/89, divergence, price positioning)  
- 📰 **Stock news risk assessment** agent  
- 🧾 **PDP (KAP) filing analysis** agent  
- 🤖 **Supervisor agent** to generate consolidated investment reports  
- 🌐 RESTful API powered by **FastAPI**  
- 🐳 **GPU-enabled Docker Compose** deployment  

---

## 🧱 Architecture Overview

```yaml
services:
  - ollama: LLM model server (GPU-enabled)
  - api: FastAPI backend for analysis and strategy
```

---

## 📂 Project Structure
```
.
├── docker-compose.yml
├── ollama/                   # Ollama model setup
├── api/
│   ├── main.py               # FastAPI app
│   ├── agents.py             # LangGraph agents
│   └── tools/                # Custom tools for analysis

```

![alt text](./documents/project_plan.svg)
---
## ⚙️ Setup
`Requirements:`
- Docker
- NVIDIA Container Toolkit (for GPU usage)

`Launch:`
```
docker compose up --build
```
Once running:

Llama3 model is pulled via /pull-llama3.sh

FastAPI is served at `http://localhost:8080`

Ollama model server is accessible at `http://localhost:11434`

---

## 📡 API Endpoints
### `GET /`
-Health check endpoint.

```json
{ "message": "🎉 FastAPI working!" }
```
### `GET /get_insights?ticker=XYZ.IS`

Returns a full investment report using AI agents.

Each report includes:

- 📊 Technical analysis (RSI, EMA34, EMA89, etc.)

- ⚠️ Risk score from recent financial news

- 🧾 PDP filing summary and market impact

- ✅ Actionable investment recommendations (short, mid, long term)

```json
📊 Ticker: THYAO.IS
🔍 Technical Summary:
Signal: Neutral

EMA34: ₺319.30

EMA89: ₺317.47

Price Above EMA34: No

RSI-Fibonacci Divergence: No

Channel Position: Bottom

📈 Investment Strategy Suggestion:
Monitor for any bullish signals such as a crossover above EMA34, which could indicate a shift towards a more positive trend and an opportunity to enter a long position.

Set stop-loss just below the current low to manage potential losses effectively.

⚠ Risk Analysis:
Risk Score: 4/10

Reasoning: The neutral RSI suggests no immediate downward pressure. However, a break above EMA34 could signal an uptrend. Broader market conditions and fundamental factors should be carefully considered before action.

🧠 Investment Strategy Plan:
Short-term Trading:

Utilize short-term strategies such as day trading or swing trading to capitalize on potential volatility around holidays.

Long Position:

Enter a long position if bullish confirmation occurs, with stop-loss and take-profit levels adjusted according to your risk tolerance.

Monitor for Breakouts:

Watch for any breakout from the current channel, especially if supported by positive news (e.g., increased passenger traffic during holiday seasons).

🔎 Always conduct thorough fundamental and technical analysis before making any investment decisions.

```
### `GET /stock_price?ticker_list=XYZ.IS,ABC.IS`
- Returns RSI-based technical indicators for the given stock.

---

## 🧠 Tech Stack
- `FastAPI` – Modern web API framework

- `LangChain + LangGraph` – Multi-agent orchestration

- `ChatOllama (Llama 3.1)` – Language model server

- `Docker + NVIDIA Runtime` – Containerized GPU support

- `Custom Tools` – Technical analysis, Yahoo Finance news scraper, PDP scraper

---

## 🧪 Example Usage

```bash
curl "http://localhost:8080/get_insights?ticker_list=ARCLK.IS,VESTL.IS"
```

---
📌 Notes for Developers
- Initial Llama3 model pull may take time.

- Ollama requires NVIDIA GPU (no CPU fallback supported).

- LangGraph supervisor merges insights from all agents into a final decision.

---
## 🛠 Contributing
- Feel free to contribute!

---
## 📄 License
- This project is licensed under the MIT License.