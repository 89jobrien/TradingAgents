import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from io import StringIO

from tradingagents.agents.discovery.models import (
    DiscoveryResult,
    DiscoveryRequest,
    DiscoveryStatus,
    TrendingStock,
    NewsArticle,
    Sector,
    EventCategory,
)


@pytest.fixture
def sample_trending_stocks():
    """
    Create a list of sample TrendingStock instances used by tests.
    
    Each TrendingStock is populated with realistic test data (ticker, company_name, score, mention_count,
    sentiment, sector, event_type, news_summary) and references a shared NewsArticle in `source_articles`.
    
    Returns:
        list[TrendingStock]: Three sample TrendingStock objects for AAPL, MSFT, and NVDA.
    """
    article = NewsArticle(
        title="Apple announces new iPhone",
        source="Reuters",
        url="https://reuters.com/article",
        published_at=datetime.now(),
        content_snippet="Apple Inc. unveiled its latest iPhone model today...",
        ticker_mentions=["AAPL"],
    )
    return [
        TrendingStock(
            ticker="AAPL",
            company_name="Apple Inc.",
            score=8.5,
            mention_count=10,
            sentiment=0.7,
            sector=Sector.TECHNOLOGY,
            event_type=EventCategory.PRODUCT_LAUNCH,
            news_summary="Apple announced new iPhone model with enhanced AI capabilities.",
            source_articles=[article],
        ),
        TrendingStock(
            ticker="MSFT",
            company_name="Microsoft Corporation",
            score=7.2,
            mention_count=8,
            sentiment=0.5,
            sector=Sector.TECHNOLOGY,
            event_type=EventCategory.EARNINGS,
            news_summary="Microsoft reported strong quarterly earnings.",
            source_articles=[article],
        ),
        TrendingStock(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            score=6.8,
            mention_count=6,
            sentiment=0.4,
            sector=Sector.TECHNOLOGY,
            event_type=EventCategory.PRODUCT_LAUNCH,
            news_summary="NVIDIA unveiled new AI chips.",
            source_articles=[article],
        ),
    ]


@pytest.fixture
def sample_discovery_result(sample_trending_stocks):
    """
    Builds a completed DiscoveryResult populated with the given trending stocks and a preset 24h request.
    
    Parameters:
        sample_trending_stocks (list[TrendingStock]): TrendingStock instances to include in the result.
    
    Returns:
        DiscoveryResult: A DiscoveryResult with a DiscoveryRequest (lookback_period="24h", max_results=20), status set to `DiscoveryStatus.COMPLETED`, `trending_stocks` set to the provided list, and `started_at`/`completed_at` set to the current time.
    """
    request = DiscoveryRequest(
        lookback_period="24h",
        max_results=20,
    )
    return DiscoveryResult(
        request=request,
        trending_stocks=sample_trending_stocks,
        status=DiscoveryStatus.COMPLETED,
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )


class TestDiscoveryMenuOption:
    def test_discover_trending_flow_exists(self):
        from cli.main import discover_trending_flow
        assert callable(discover_trending_flow)

    def test_select_lookback_period_function_exists(self):
        from cli.main import select_lookback_period
        assert callable(select_lookback_period)


class TestLookbackSelection:
    @patch("cli.main.questionary.select")
    def test_lookback_selection_returns_valid_period(self, mock_select):
        mock_select.return_value.ask.return_value = "24h"
        from cli.main import select_lookback_period
        result = select_lookback_period()
        assert result in ["1h", "6h", "24h", "7d"]

    @patch("cli.main.questionary.select")
    def test_lookback_selection_handles_all_options(self, mock_select):
        from cli.main import select_lookback_period
        for period in ["1h", "6h", "24h", "7d"]:
            mock_select.return_value.ask.return_value = period
            result = select_lookback_period()
            assert result == period


class TestResultsTableDisplay:
    def test_create_discovery_results_table(self, sample_trending_stocks):
        from cli.main import create_discovery_results_table
        table = create_discovery_results_table(sample_trending_stocks)
        assert table is not None
        assert table.row_count == len(sample_trending_stocks)

    def test_table_has_correct_columns(self, sample_trending_stocks):
        from cli.main import create_discovery_results_table
        table = create_discovery_results_table(sample_trending_stocks)
        column_names = [col.header for col in table.columns]
        expected_columns = ["Rank", "Ticker", "Company", "Score", "Mentions", "Event Type"]
        for expected in expected_columns:
            assert expected in column_names


class TestDetailView:
    def test_create_stock_detail_panel(self, sample_trending_stocks):
        from cli.main import create_stock_detail_panel
        stock = sample_trending_stocks[0]
        panel = create_stock_detail_panel(stock, rank=1)
        assert panel is not None