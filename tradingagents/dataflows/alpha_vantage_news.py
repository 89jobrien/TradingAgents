import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from .alpha_vantage_common import _make_api_request, format_datetime_for_api

def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """
    Retrieve news sentiment for a ticker within a specified date range.
    
    Parameters:
        ticker (str): Ticker symbol or comma-separated list of ticker symbols to query.
        start_date (datetime | str): Start of the time window (datetime or string accepted; will be formatted for the API).
        end_date (datetime | str): End of the time window (datetime or string accepted; will be formatted for the API).
    
    Returns:
        dict[str, str] | str: The parsed API response as a dictionary when available, otherwise the raw response string.
    """
    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
        "sort": "LATEST",
        "limit": "50",
    }

    return _make_api_request("NEWS_SENTIMENT", params)

def get_insider_transactions(symbol: str) -> dict[str, str] | str:
    """
    Retrieve insider transaction data for the given symbol from the Alpha Vantage API.
    
    Parameters:
        symbol (str): The ticker symbol to query for insider transactions.
    
    Returns:
        dict[str, str]: Parsed API response containing insider transaction data when successful,
        or str: raw error message or response string when the request did not return a dict.
    """
    params = {
        "symbol": symbol,
    }

    return _make_api_request("INSIDER_TRANSACTIONS", params)


def get_bulk_news_alpha_vantage(lookback_hours: int) -> List[Dict[str, Any]]:
    """
    Retrieve recent news articles from Alpha Vantage for a lookback window.
    
    Query the NEWS_SENTIMENT endpoint for the past `lookback_hours` hours and return a normalized list of article dictionaries. Items that cannot be parsed or produce processing errors are skipped; if the API response is invalid or cannot be decoded, an empty list is returned.
    
    Parameters:
        lookback_hours (int): Number of hours to look back from the current time.
    
    Returns:
        List[dict]: A list of articles where each dictionary contains:
            - "title" (str): Article title (empty string if missing).
            - "source" (str): Source name (empty string if missing).
            - "url" (str): Article URL (empty string if missing).
            - "published_at" (str): Publication timestamp in ISO 8601 format.
            - "content_snippet" (str): Up to the first 500 characters of the article summary.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=lookback_hours)

    params = {
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
        "sort": "LATEST",
        "limit": "200",
        "topics": "financial_markets,earnings,economy_fiscal,economy_monetary,mergers_and_acquisitions",
    }

    response = _make_api_request("NEWS_SENTIMENT", params)

    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return []

    if not isinstance(response, dict):
        return []

    feed = response.get("feed", [])

    articles = []
    for item in feed:
        try:
            time_published = item.get("time_published", "")
            if time_published:
                try:
                    published_at = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                except ValueError:
                    published_at = datetime.now()
            else:
                published_at = datetime.now()

            article = {
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "published_at": published_at.isoformat(),
                "content_snippet": item.get("summary", "")[:500],
            }
            articles.append(article)
        except Exception:
            continue

    return articles