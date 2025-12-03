from typing import Annotated, List, Dict, Any
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from .googlenews_utils import getNewsData


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """
    Fetches Google News results for a query over a past date window and returns them as a markdown-formatted string.
    
    Parameters:
        query (str): Search query; spaces are replaced with plus signs for the request.
        curr_date (str): End date in "YYYY-MM-DD" format.
        look_back_days (int): Number of days to look back from `curr_date` to form the start of the window.
    
    Returns:
        news_markdown (str): A markdown string with a header showing the query and date range and each article as:
            "### {title} (source: {source})" followed by the snippet. Returns an empty string if no results were found.
    """
    query = query.replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    news_results = getNewsData(query, before, curr_date)

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {query} Google News, from {before} to {curr_date}:\n\n{news_str}"


def get_bulk_news_google(lookback_hours: int) -> List[Dict[str, Any]]:
    """
    Collect recent Google News articles for a fixed set of finance-related queries and return deduplicated article records.
    
    Queries the Google News feed for "stock market", "trading news", and "earnings report" over the past lookback_hours, deduplicates results by title, and returns a list of article dictionaries. If a query fails, it is skipped; published timestamps default to the current time when not provided or parseable.
    
    Parameters:
        lookback_hours (int): Number of hours to look back from the current time to fetch articles.
    
    Returns:
        List[Dict[str, Any]]: A list of article dictionaries with the keys:
            - "title": article title
            - "source": article source (defaults to "Google News")
            - "url": link to the article (empty string if unavailable)
            - "published_at": ISO 8601 timestamp string of publication (may default to current time)
            - "content_snippet": up to the first 500 characters of the article snippet
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=lookback_hours)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    queries = [
        "stock market",
        "trading news",
        "earnings report",
    ]

    all_articles = []
    seen_titles = set()

    for query in queries:
        try:
            news_results = getNewsData(query.replace(" ", "+"), start_str, end_str)

            for news in news_results:
                title = news.get("title", "")
                if title and title not in seen_titles:
                    seen_titles.add(title)

                    date_str = news.get("date", "")
                    try:
                        if date_str:
                            published_at = datetime.now()
                        else:
                            published_at = datetime.now()
                    except ValueError:
                        published_at = datetime.now()

                    article = {
                        "title": title,
                        "source": news.get("source", "Google News"),
                        "url": news.get("link", ""),
                        "published_at": published_at.isoformat(),
                        "content_snippet": news.get("snippet", "")[:500],
                    }
                    all_articles.append(article)

        except Exception:
            continue

    return all_articles