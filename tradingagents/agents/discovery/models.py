from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any


class DiscoveryStatus(Enum):
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Sector(Enum):
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    ENERGY = "energy"
    CONSUMER_GOODS = "consumer_goods"
    INDUSTRIALS = "industrials"
    OTHER = "other"


class EventCategory(Enum):
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    REGULATORY = "regulatory"
    PRODUCT_LAUNCH = "product_launch"
    EXECUTIVE_CHANGE = "executive_change"
    OTHER = "other"


@dataclass
class NewsArticle:
    title: str
    source: str
    url: str
    published_at: datetime
    content_snippet: str
    ticker_mentions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the NewsArticle to a JSON-serializable dictionary.
        
        Returns:
            dict: Mapping with keys "title", "source", "url", "published_at" (ISO 8601 string), "content_snippet", and "ticker_mentions".
        """
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "content_snippet": self.content_snippet,
            "ticker_mentions": self.ticker_mentions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NewsArticle":
        """
        Constructs a NewsArticle from a dictionary containing serialized article fields.
        
        Parameters:
            data (Dict[str, Any]): Mapping with keys:
                - "title": article title string
                - "source": source name string
                - "url": article URL string
                - "published_at": ISO-format datetime string
                - "content_snippet": short content string
                - "ticker_mentions": list of ticker symbol strings
        
        Returns:
            news_article (NewsArticle): NewsArticle instance built from the provided data.
        """
        return cls(
            title=data["title"],
            source=data["source"],
            url=data["url"],
            published_at=datetime.fromisoformat(data["published_at"]),
            content_snippet=data["content_snippet"],
            ticker_mentions=data["ticker_mentions"],
        )


@dataclass
class TrendingStock:
    ticker: str
    company_name: str
    score: float
    mention_count: int
    sentiment: float
    sector: Sector
    event_type: EventCategory
    news_summary: str
    source_articles: List[NewsArticle]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the TrendingStock into a JSON-serializable dictionary.
        
        Returns:
            dict: Dictionary with the trending stock's fields. Enum fields `sector` and `event_type` are converted to their string values, and `source_articles` is a list of each article serialized via its `to_dict()` method.
        """
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "score": self.score,
            "mention_count": self.mention_count,
            "sentiment": self.sentiment,
            "sector": self.sector.value,
            "event_type": self.event_type.value,
            "news_summary": self.news_summary,
            "source_articles": [article.to_dict() for article in self.source_articles],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrendingStock":
        """
        Constructs a TrendingStock instance from its dictionary representation.
        
        Parameters:
            data (Dict[str, Any]): Dictionary with keys "ticker", "company_name", "score", "mention_count",
                "sentiment", "sector" (sector name string), "event_type" (event category name string),
                "news_summary", and "source_articles" (list of article dicts compatible with NewsArticle.from_dict).
        
        Returns:
            TrendingStock: A TrendingStock built from the provided dictionary.
        """
        return cls(
            ticker=data["ticker"],
            company_name=data["company_name"],
            score=data["score"],
            mention_count=data["mention_count"],
            sentiment=data["sentiment"],
            sector=Sector(data["sector"]),
            event_type=EventCategory(data["event_type"]),
            news_summary=data["news_summary"],
            source_articles=[
                NewsArticle.from_dict(article) for article in data["source_articles"]
            ],
        )


@dataclass
class DiscoveryRequest:
    lookback_period: str
    sector_filter: Optional[List[Sector]] = None
    event_filter: Optional[List[EventCategory]] = None
    max_results: int = 20
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the DiscoveryRequest to a JSON-serializable dictionary.
        
        The returned dictionary contains:
        - "lookback_period": the lookback period string.
        - "sector_filter": a list of sector names as strings or `None` if no filter is set.
        - "event_filter": a list of event category names as strings or `None` if no filter is set.
        - "max_results": the maximum number of results.
        - "created_at": the creation timestamp as an ISO 8601 string.
        
        Returns:
            dict: A mapping suitable for JSON serialization representing this request.
        """
        return {
            "lookback_period": self.lookback_period,
            "sector_filter": (
                [s.value for s in self.sector_filter] if self.sector_filter else None
            ),
            "event_filter": (
                [e.value for e in self.event_filter] if self.event_filter else None
            ),
            "max_results": self.max_results,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveryRequest":
        """
        Constructs a DiscoveryRequest from a mapping, converting string values to their corresponding enums and parsing timestamps.
        
        Parameters:
            data (Dict[str, Any]): Dictionary with keys:
                - "lookback_period" (str): Lookback period identifier.
                - "sector_filter" (Optional[List[str]]): Optional list of sector names to convert to Sector enum values.
                - "event_filter" (Optional[List[str]]): Optional list of event category names to convert to EventCategory enum values.
                - "max_results" (Optional[int]): Maximum number of results; defaults to 20 if absent.
                - "created_at" (str): ISO-format datetime string for the request creation time.
        
        Returns:
            DiscoveryRequest: Instance populated from the provided mapping; `sector_filter` and `event_filter` are lists of enums or None, and `created_at` is parsed from the ISO-format string.
        """
        return cls(
            lookback_period=data["lookback_period"],
            sector_filter=(
                [Sector(s) for s in data["sector_filter"]]
                if data.get("sector_filter")
                else None
            ),
            event_filter=(
                [EventCategory(e) for e in data["event_filter"]]
                if data.get("event_filter")
                else None
            ),
            max_results=data.get("max_results", 20),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class DiscoveryResult:
    request: DiscoveryRequest
    trending_stocks: List[TrendingStock]
    status: DiscoveryStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the DiscoveryResult into a JSON-serializable dictionary.
        
        Produces a dictionary containing the serialized discovery request, list of serialized trending stocks, the status as a string, ISO-formatted start and (optional) completion timestamps, and any error message.
        
        Returns:
            dict: Mapping with keys:
                - "request": serialized DiscoveryRequest
                - "trending_stocks": list of serialized TrendingStock objects
                - "status": discovery status as a string
                - "started_at": ISO 8601 string of the start time
                - "completed_at": ISO 8601 string of the completion time or `None`
                - "error_message": error message string or `None`
        """
        return {
            "request": self.request.to_dict(),
            "trending_stocks": [stock.to_dict() for stock in self.trending_stocks],
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveryResult":
        """
        Create a DiscoveryResult from its dictionary representation.
        
        Expects a dict produced by DiscoveryResult.to_dict(), with keys:
        `request` (dict), `trending_stocks` (list of dicts), `status` (str),
        `started_at` (ISO datetime string), optional `completed_at` (ISO datetime string or None),
        and optional `error_message` (str).
        
        Parameters:
            data (Dict[str, Any]): Dictionary containing DiscoveryResult fields as produced by to_dict().
        
        Returns:
            DiscoveryResult: An instance populated from the provided dictionary.
        """
        return cls(
            request=DiscoveryRequest.from_dict(data["request"]),
            trending_stocks=[
                TrendingStock.from_dict(stock) for stock in data["trending_stocks"]
            ],
            status=DiscoveryStatus(data["status"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            error_message=data.get("error_message"),
        )