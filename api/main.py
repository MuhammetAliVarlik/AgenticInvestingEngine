from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agents import ask
from api.tools.stockPriceAnaliserTool import rsi_predictor
 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify the origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "🎉 FastAPI working!"}

@app.get("/stock_price")
async def get_insights(ticker: str):
    return rsi_predictor(ticker)


@app.get("/get_insights")
async def get_insights(ticker_list: str):
    question = ticker_list + " make an investment strategy to maximize profit to this stocks."
    return ask(question=question)
