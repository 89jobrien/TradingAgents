import os
import signal
import threading
from pathlib import Path
import json
from datetime import date, datetime
from typing import Dict, Any, Tuple, List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_sentiment,
    get_insider_transactions,
    get_global_news
)

from tradingagents.agents.discovery import (
    DiscoveryRequest,
    DiscoveryResult,
    DiscoveryStatus,
    TrendingStock,
    Sector,
    EventCategory,
    DiscoveryTimeoutError,
    extract_entities,
    calculate_trending_scores,
)
from tradingagents.dataflows.interface import get_bulk_news

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class DiscoveryTimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    """
    Signal handler that raises a DiscoveryTimeoutException to indicate the discovery operation exceeded its allowed time.
    
    Parameters:
        signum: The signal number received.
        frame: The current stack frame (or None) at the time the signal was received.
    
    Raises:
        DiscoveryTimeoutException: Always raised to abort a timed-out discovery operation.
    """
    raise DiscoveryTimeoutException("Discovery operation timed out")


class TradingAgentsGraph:

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize a TradingAgentsGraph instance with LLMs, memories, tools, and graph components wired together.
        
        Constructs and configures LLM clients based on `config["llm_provider"]`, creates financial situation memories, builds tool nodes, sets up conditional logic and the agent graph, and prepares supporting components for propagation, reflection, and signal processing.
        
        Parameters:
            selected_analysts (list[str]): List of analyst categories to include in the graph (defaults to ["market", "social", "news", "fundamentals"]).
            debug (bool): If True, enables debug behaviour for graph execution and streaming (default False).
            config (dict | None): Configuration dictionary overriding defaults; used for project paths, LLM provider and model names, backend URLs, and other runtime options. If None, DEFAULT_CONFIG is used.
        
        Raises:
            ValueError: If `config["llm_provider"]` is not one of the supported providers ("openai", "ollama", "openrouter", "anthropic", "google").
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG

        set_config(self.config)

        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama" or self.config["llm_provider"] == "openrouter":
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "anthropic":
            self.deep_thinking_llm = ChatAnthropic(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatAnthropic(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "google":
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"])
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"])
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")

        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)
        self.tool_nodes = self._create_tool_nodes()
        self.conditional_logic = ConditionalLogic()
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """
        Builds and returns the set of categorized ToolNode objects used by the agent graph.
        
        Returns:
            tool_nodes (Dict[str, ToolNode]): Mapping of category names to ToolNode instances:
                - "market": tools for fetching stock data and indicators.
                - "social": tools for retrieving social/news mentions.
                - "news": tools for news, global news, insider sentiment, and insider transactions.
                - "fundamentals": tools for financial fundamentals and statements.
        """
        return {
            "market": ToolNode(
                [
                    get_stock_data,
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    get_news,
                    get_global_news,
                    get_insider_sentiment,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """
        Run the agent graph for a company on a given trade date and return the final state and processed trading signal.
        
        Parameters:
            company_name (str): Company ticker or identifier to analyze.
            trade_date (date): Date for which the agents should generate the trading analysis.
        
        Returns:
            tuple: A pair (final_state, processed_signal) where `final_state` is the agent graph's resulting state dictionary and `processed_signal` is the signal returned by processing `final_state["final_trade_decision"]`.
        """
        self.ticker = company_name
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            final_state = self.graph.invoke(init_agent_state, **args)

        self.curr_state = final_state
        self._log_state(trade_date, final_state)

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """
        Record and persist the agent's final state for a given trade date.
        
        Stores a structured subset of `final_state` under `self.log_states_dict[str(trade_date)]`
        and writes the entire `log_states_dict` to
        eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json,
        creating the directory if necessary.
        
        Parameters:
        	trade_date (date | str): The trade date used as the log key and filename suffix.
        	final_state (dict): The final agent state object containing keys such as
        		company_of_interest, market_report, sentiment_report, news_report,
        		fundamentals_report, investment_debate_state, trader_investment_plan,
        		risk_debate_state, investment_plan, and final_trade_decision.
        """
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """
        Update internal agent memories by reflecting on the provided returns/losses.
        
        Parameters:
            returns_losses: A value or structure representing recent returns and losses (e.g., profit/loss metrics) used to update each role-specific memory (bull researcher, bear researcher, trader, investment judge, and risk manager).
        """
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """
        Convert a raw final trade-decision signal into a normalized actionable signal.
        
        Parameters:
        	full_signal (Any): The raw full decision produced by the agents (typically a dict or object containing decision details, rationale, and metadata).
        
        Returns:
        	Any: A normalized signal suitable for execution or downstream processing (for example an action string or structured signal object).
        """
        return self.signal_processor.process_signal(full_signal)

    def discover_trending(
        self,
        request: Optional[DiscoveryRequest] = None,
    ) -> DiscoveryResult:
        """
        Discover trending stocks from recent news within a configurable lookback period.
        
        Parameters:
            request (Optional[DiscoveryRequest]): Discovery parameters (lookback_period, max_results, sector_filter, event_filter).
                If omitted, a default request is created with a 24h lookback and `discovery_max_results` from config (default 20).
        
        Returns:
            DiscoveryResult: Result object containing the original request, the list of discovered trending stocks (possibly filtered
            by sector or event), discovery status (PROCESSING/COMPLETED/FAILED), start and completion timestamps, and an error_message
            when applicable.
        
        Raises:
            DiscoveryTimeoutError: If the discovery operation exceeds the configured hard timeout (`discovery_hard_timeout` in config).
        """
        if request is None:
            request = DiscoveryRequest(
                lookback_period="24h",
                max_results=self.config.get("discovery_max_results", 20),
            )

        started_at = datetime.now()
        result = DiscoveryResult(
            request=request,
            trending_stocks=[],
            status=DiscoveryStatus.PROCESSING,
            started_at=started_at,
        )

        hard_timeout = self.config.get("discovery_hard_timeout", 120)

        discovery_result = {"stocks": [], "error": None}

        def run_discovery():
            """
            Populate the outer discovery_result dictionary with trending stocks computed from recent news articles or record an error.
            
            Fetches articles for request.lookback_period, extracts entity mentions using self.config, computes trending scores constrained by the `discovery_min_mentions` and `discovery_max_results` configuration (overridden by `request.max_results` when provided), and assigns the resulting list to `discovery_result["stocks"]`. On exception, stores the exception string in `discovery_result["error"]`.
            """
            try:
                articles = get_bulk_news(request.lookback_period)

                mentions = extract_entities(articles, self.config)

                min_mentions = self.config.get("discovery_min_mentions", 2)
                max_results = request.max_results or self.config.get("discovery_max_results", 20)

                trending_stocks = calculate_trending_scores(
                    mentions,
                    articles,
                    max_results=max_results,
                    min_mentions=min_mentions,
                )

                discovery_result["stocks"] = trending_stocks
            except Exception as e:
                discovery_result["error"] = str(e)

        discovery_thread = threading.Thread(target=run_discovery)
        discovery_thread.start()
        discovery_thread.join(timeout=hard_timeout)

        if discovery_thread.is_alive():
            raise DiscoveryTimeoutError(
                f"Discovery operation exceeded {hard_timeout} second timeout"
            )

        if discovery_result["error"]:
            result.status = DiscoveryStatus.FAILED
            result.error_message = discovery_result["error"]
            result.completed_at = datetime.now()
            return result

        trending_stocks = discovery_result["stocks"]

        if request.sector_filter:
            sector_values = {s.value if isinstance(s, Sector) else s for s in request.sector_filter}
            trending_stocks = [
                stock for stock in trending_stocks
                if stock.sector.value in sector_values or stock.sector in request.sector_filter
            ]

        if request.event_filter:
            event_values = {e.value if isinstance(e, EventCategory) else e for e in request.event_filter}
            trending_stocks = [
                stock for stock in trending_stocks
                if stock.event_type.value in event_values or stock.event_type in request.event_filter
            ]

        result.trending_stocks = trending_stocks
        result.status = DiscoveryStatus.COMPLETED
        result.completed_at = datetime.now()

        return result

    def analyze_trending(
        self,
        trending_stock: TrendingStock,
        trade_date: Optional[date] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Analyze a single trending stock by running the trading agents graph for a given trade date.
        
        Parameters:
            trending_stock (TrendingStock): TrendingStock object whose `ticker` will be analyzed.
            trade_date (date, optional): Date for which to run the analysis; defaults to today's date.
        
        Returns:
            Tuple[Dict[str, Any], str]: A tuple with the final agent state dictionary and the processed trade signal string.
        """
        ticker = trending_stock.ticker

        if trade_date is None:
            trade_date = date.today()

        return self.propagate(ticker, trade_date)