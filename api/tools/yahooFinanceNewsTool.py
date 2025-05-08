from langchain_core.tools import tool
import yfinance as yf

@tool
def stock_news(ticker: str) -> list:
    """
    Retrieve the latest news articles discussing a particular stock ticker.
    Parameters:
        - ticker: Stock ticker symbol.
    Returns:
        - A list of dictionaries containing limited news articles.
    """
    max_articles = 5
    # Get the ticker object
    ticker_obj = yf.Ticker(ticker)
    news = ticker_obj.get_news()  # Fetch news articles

    # Ensure `news` is a list and limit the articles to max_articles
    if isinstance(news, list):
        selected_news = news[:max_articles]  # Limit the articles
        
        # Process and return formatted articles
        formatted_news = []
        for article in selected_news:
            article = article.get("content")
            formatted_article = {
                "Summary": article.get("summary", "Summary not available"),
            }
            formatted_news.append(formatted_article)

        return formatted_news

    return []