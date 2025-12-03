from typing import Optional, List
import datetime
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule
import questionary

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.discovery.models import (
    DiscoveryRequest,
    DiscoveryResult,
    DiscoveryStatus,
    TrendingStock,
    Sector,
    EventCategory,
)
from tradingagents.agents.discovery.persistence import save_discovery_result
from cli.models import AnalystType
from cli.utils import *

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,
)


class MessageBuffer:
    def __init__(self, max_length=100):
        """
        Initialize the message and report buffer used to collect streaming messages, tool calls, agent statuses, and per-section reports.
        
        Parameters:
            max_length (int): Maximum number of recent messages and tool calls to retain in the buffer.
        """
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None
        self.agent_status = {
            "Market Analyst": "pending",
            "Social Analyst": "pending",
            "News Analyst": "pending",
            "Fundamentals Analyst": "pending",
            "Bull Researcher": "pending",
            "Bear Researcher": "pending",
            "Research Manager": "pending",
            "Trader": "pending",
            "Risky Analyst": "pending",
            "Neutral Analyst": "pending",
            "Safe Analyst": "pending",
            "Portfolio Manager": "pending",
        }
        self.current_agent = None
        self.report_sections = {
            "market_report": None,
            "sentiment_report": None,
            "news_report": None,
            "fundamentals_report": None,
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None,
        }

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        """
        Store content for a named report section and refresh the current and final reports.
        
        Parameters:
            section_name (str): Key identifying a report section; must be one of the keys in self.report_sections (e.g., 'market_report', 'sentiment_report', 'news_report', 'fundamentals_report', 'investment_plan', 'trader_investment_plan', 'final_trade_decision').
            content (str | None): The textual content to save for the section; may be None to clear the section.
        """
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        """
        Update the MessageBuffer's current_report to reflect the most recently populated report section and refresh the aggregated final_report.
        
        Finds the last non-None entry in self.report_sections, maps that section key to a human-readable heading, and sets self.current_report to a markdown-formatted heading plus the section content. Also invokes self._update_final_report() to recompute the consolidated final report.
        """
        latest_section = None
        latest_content = None

        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content

        if latest_section and latest_content:
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        self._update_final_report()

    def _update_final_report(self):
        """
        Builds the aggregated final report from available report sections and assigns it to self.final_report.
        
        The method composes the final report by appending present sections in a fixed order with headings:
        - Analyst Team Reports (includes market, sentiment, news, and fundamentals subsections when present)
        - Research Team Decision (investment_plan)
        - Trading Team Plan (trader_investment_plan)
        - Portfolio Management Decision (final_trade_decision)
        
        If no report sections are available, self.final_report is set to None.
        """
        report_parts = []

        if any(
            self.report_sections[section]
            for section in [
                "market_report",
                "sentiment_report",
                "news_report",
                "fundamentals_report",
            ]
        ):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections["market_report"]:
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections["sentiment_report"]:
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections["news_report"]:
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections["fundamentals_report"]:
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        if self.report_sections["investment_plan"]:
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        if self.report_sections["trader_investment_plan"]:
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        if self.report_sections["final_trade_decision"]:
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


LOOKBACK_OPTIONS = [
    ("Last hour (1h)", "1h"),
    ("Last 6 hours (6h)", "6h"),
    ("Last 24 hours (24h)", "24h"),
    ("Last 7 days (7d)", "7d"),
]

SECTOR_OPTIONS = [
    ("Technology", Sector.TECHNOLOGY),
    ("Healthcare", Sector.HEALTHCARE),
    ("Finance", Sector.FINANCE),
    ("Energy", Sector.ENERGY),
    ("Consumer Goods", Sector.CONSUMER_GOODS),
    ("Industrials", Sector.INDUSTRIALS),
    ("Other", Sector.OTHER),
]

EVENT_OPTIONS = [
    ("Earnings", EventCategory.EARNINGS),
    ("Merger/Acquisition", EventCategory.MERGER_ACQUISITION),
    ("Regulatory", EventCategory.REGULATORY),
    ("Product Launch", EventCategory.PRODUCT_LAUNCH),
    ("Executive Change", EventCategory.EXECUTIVE_CHANGE),
    ("Other", EventCategory.OTHER),
]


def create_question_box(title: str, prompt: str, default: str = None) -> Panel:
    """
    Create a styled Rich Panel used as a prompt/question box.
    
    Parameters:
        title (str): Header text displayed in bold at the top of the panel.
        prompt (str): Instructional or descriptive text shown under the title.
        default (str, optional): Optional default value shown in dim text beneath the prompt.
    
    Returns:
        Panel: A Rich Panel containing the formatted title, prompt, and optional default, styled with a blue border and padding.
    """
    box_content = f"[bold]{title}[/bold]\n"
    box_content += f"[dim]{prompt}[/dim]"
    if default:
        box_content += f"\n[dim]Default: {default}[/dim]"
    return Panel(box_content, border_style="blue", padding=(1, 2))


def select_lookback_period() -> str:
    """
    Prompt the user to choose a lookback period for discovery.
    
    Prompts with interactive choices defined by LOOKBACK_OPTIONS. If the user cancels or makes no selection, the process exits with status code 1.
    
    Returns:
        str: The selected lookback period value.
    """
    choice = questionary.select(
        "Select lookback period:",
        choices=[
            questionary.Choice(display, value=value) for display, value in LOOKBACK_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "fg:cyan noinherit"),
                ("pointer", "fg:cyan noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No lookback period selected. Exiting...[/red]")
        exit(1)

    return choice


def select_sector_filter() -> Optional[List[Sector]]:
    """
    Prompt the user to optionally choose one or more sectors to filter discovery results.
    
    Returns:
        A list of selected `Sector` values when the user enables filtering and selects at least one sector, or `None` if the user declines to filter or makes no selection.
    """
    use_filter = questionary.confirm(
        "Filter by sector?",
        default=False,
        style=questionary.Style(
            [
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "fg:cyan noinherit"),
            ]
        ),
    ).ask()

    if not use_filter:
        return None

    choices = questionary.checkbox(
        "Select sectors to include:",
        choices=[
            questionary.Choice(display, value=value) for display, value in SECTOR_OPTIONS
        ],
        instruction="\n- Press Space to select/unselect\n- Press 'a' to select all\n- Press Enter when done",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:cyan"),
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        return None

    return choices


def select_event_filter() -> Optional[List[EventCategory]]:
    """
    Prompt the user to optionally choose one or more event categories to filter discovery results.
    
    Returns:
        Optional[List[EventCategory]]: A list of selected `EventCategory` values when the user enables filtering and picks at least one option, or `None` if the user declines to filter or selects no choices.
    """
    use_filter = questionary.confirm(
        "Filter by event type?",
        default=False,
        style=questionary.Style(
            [
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "fg:cyan noinherit"),
            ]
        ),
    ).ask()

    if not use_filter:
        return None

    choices = questionary.checkbox(
        "Select event types to include:",
        choices=[
            questionary.Choice(display, value=value) for display, value in EVENT_OPTIONS
        ],
        instruction="\n- Press Space to select/unselect\n- Press 'a' to select all\n- Press Enter when done",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:cyan"),
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        return None

    return choices


def create_discovery_results_table(trending_stocks: List[TrendingStock]) -> Table:
    """
    Builds a Rich Table listing trending stocks and their key metrics.
    
    Parameters:
        trending_stocks (List[TrendingStock]): Ordered list of trending stock records. Each record is expected to provide
            `ticker`, `company_name`, `score`, `mention_count`, and `event_type`.
    
    Returns:
        Table: A Rich Table populated with rows for each trending stock and the columns
        "Rank", "Ticker", "Company", "Score", "Mentions", and "Event Type". The top three ranks and their tickers
        are styled for emphasis; company names are truncated to fit the table width.
    """
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        title="Trending Stocks",
        title_style="bold green",
        expand=True,
    )

    table.add_column("Rank", style="cyan", justify="center", width=6)
    table.add_column("Ticker", style="bold yellow", justify="center", width=10)
    table.add_column("Company", style="white", justify="left", width=25)
    table.add_column("Score", style="green", justify="right", width=10)
    table.add_column("Mentions", style="blue", justify="center", width=10)
    table.add_column("Event Type", style="magenta", justify="center", width=18)

    for rank, stock in enumerate(trending_stocks, 1):
        if rank <= 3:
            rank_display = f"[bold green]{rank}[/bold green]"
            ticker_display = f"[bold yellow]{stock.ticker}[/bold yellow]"
        else:
            rank_display = str(rank)
            ticker_display = stock.ticker

        table.add_row(
            rank_display,
            ticker_display,
            stock.company_name[:25] if len(stock.company_name) > 25 else stock.company_name,
            f"{stock.score:.2f}",
            str(stock.mention_count),
            stock.event_type.value.replace("_", " ").title(),
        )

    return table


def create_stock_detail_panel(stock: TrendingStock, rank: int) -> Panel:
    """
    Create a Rich Panel showing detailed information for a trending stock.
    
    Parameters:
        stock (TrendingStock): The trending stock to display; expected to provide
            ticker, company_name, score, sentiment, sector, event_type,
            mention_count, news_summary, and source_articles.
        rank (int): The stock's ranking position to display in the panel.
    
    Returns:
        Panel: A Rich Panel containing a formatted summary of the stock (rank, ticker,
        company, score, sentiment with color-coded label, sector, event type, mentions,
        a news summary, and up to the top three source articles).
    """
    sentiment_label = "positive" if stock.sentiment > 0.3 else "negative" if stock.sentiment < -0.3 else "neutral"
    sentiment_color = "green" if stock.sentiment > 0.3 else "red" if stock.sentiment < -0.3 else "yellow"

    content = f"""[bold]Rank #{rank}: {stock.ticker} - {stock.company_name}[/bold]

[cyan]Score:[/cyan] {stock.score:.2f}
[cyan]Sentiment:[/cyan] [{sentiment_color}]{stock.sentiment:.2f} ({sentiment_label})[/{sentiment_color}]
[cyan]Sector:[/cyan] {stock.sector.value.replace("_", " ").title()}
[cyan]Event Type:[/cyan] {stock.event_type.value.replace("_", " ").title()}
[cyan]Mentions:[/cyan] {stock.mention_count}

[bold]News Summary:[/bold]
{stock.news_summary}

[bold]Top Source Articles:[/bold]"""

    for i, article in enumerate(stock.source_articles[:3], 1):
        content += f"\n  {i}. [{article.title[:50]}...] - {article.source}"

    return Panel(
        content,
        title=f"Stock Details: {stock.ticker}",
        border_style="cyan",
        padding=(1, 2),
    )


def select_stock_for_detail(trending_stocks: List[TrendingStock]) -> Optional[TrendingStock]:
    """
    Present an interactive selection list of trending stocks and return the chosen stock.
    
    Parameters:
        trending_stocks (List[TrendingStock]): Ordered list of trending stocks to present to the user.
    
    Returns:
        Optional[TrendingStock]: The selected TrendingStock, or `None` if the user chooses to go back or if no stocks are provided.
    """
    if not trending_stocks:
        return None

    choices = [
        questionary.Choice(
            f"{i+1}. {stock.ticker} - {stock.company_name} (Score: {stock.score:.2f})",
            value=stock
        )
        for i, stock in enumerate(trending_stocks)
    ]
    choices.append(questionary.Choice("Back to menu", value=None))

    selected = questionary.select(
        "Select a stock to view details:",
        choices=choices,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "fg:cyan noinherit"),
                ("pointer", "fg:cyan noinherit"),
            ]
        ),
    ).ask()

    return selected


def discover_trending_flow():
    """
    Run the interactive "Discover Trending Stocks" CLI flow.
    
    Prompts the user for lookback period, optional sector and event filters, LLM provider, and model; performs a discovery request to find trending stocks, persists results when available, and displays a results table and per-stock detail panels. The flow prints progress and errors to the console, gracefully exits on failures, and allows the user to optionally start a deeper analysis for a selected ticker.
    """
    console.print(Rule("[bold green]Discover Trending Stocks[/bold green]"))
    console.print()

    console.print(
        create_question_box(
            "Step 1: Lookback Period",
            "Select how far back to search for trending stocks"
        )
    )
    lookback_period = select_lookback_period()
    console.print(f"[green]Selected lookback period:[/green] {lookback_period}")
    console.print()

    console.print(
        create_question_box(
            "Step 2: Sector Filter (Optional)",
            "Optionally filter results by sector"
        )
    )
    sector_filter = select_sector_filter()
    if sector_filter:
        console.print(f"[green]Selected sectors:[/green] {', '.join(s.value for s in sector_filter)}")
    else:
        console.print("[dim]No sector filter applied[/dim]")
    console.print()

    console.print(
        create_question_box(
            "Step 3: Event Filter (Optional)",
            "Optionally filter results by event type"
        )
    )
    event_filter = select_event_filter()
    if event_filter:
        console.print(f"[green]Selected events:[/green] {', '.join(e.value for e in event_filter)}")
    else:
        console.print("[dim]No event filter applied[/dim]")
    console.print()

    console.print(
        create_question_box(
            "Step 4: LLM Provider",
            "Select your LLM provider for entity extraction"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()
    console.print()

    console.print(
        create_question_box(
            "Step 5: Quick-Thinking Model",
            "Select the model for entity extraction"
        )
    )
    selected_model = select_shallow_thinking_agent(selected_llm_provider)
    console.print()

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = selected_llm_provider.lower()
    config["backend_url"] = backend_url
    config["quick_think_llm"] = selected_model
    config["deep_think_llm"] = selected_model

    request = DiscoveryRequest(
        lookback_period=lookback_period,
        sector_filter=sector_filter,
        event_filter=event_filter,
        max_results=config.get("discovery_max_results", 20),
    )

    discovery_stages = [
        "Fetching news...",
        "Extracting entities...",
        "Resolving tickers...",
        "Calculating scores...",
    ]

    result = None
    with Live(console=console, refresh_per_second=4) as live:
        for i, stage in enumerate(discovery_stages):
            progress_panel = Panel(
                f"[bold cyan]{stage}[/bold cyan]\n\n"
                f"[dim]Stage {i+1} of {len(discovery_stages)}[/dim]",
                title="Discovery Progress",
                border_style="cyan",
                padding=(2, 4),
            )
            live.update(Align.center(progress_panel))

            if i == 0:
                try:
                    graph = TradingAgentsGraph(config=config, debug=False)
                    result = graph.discover_trending(request)
                except Exception as e:
                    console.print(f"\n[red]Error during discovery: {e}[/red]")
                    return

            time.sleep(0.5)

    if result is None:
        console.print("\n[red]Discovery failed. Please try again.[/red]")
        return

    if result.status == DiscoveryStatus.FAILED:
        console.print(f"\n[red]Discovery failed: {result.error_message}[/red]")
        return

    if result.status == DiscoveryStatus.COMPLETED:
        try:
            save_path = save_discovery_result(result)
            console.print(f"\n[dim]Results saved to: {save_path}[/dim]")
        except Exception as e:
            console.print(f"\n[yellow]Warning: Could not save results: {e}[/yellow]")

    console.print()

    if not result.trending_stocks:
        console.print("[yellow]No trending stocks found matching your criteria.[/yellow]")
        return

    console.print(f"[green]Found {len(result.trending_stocks)} trending stocks[/green]")
    console.print()

    results_table = create_discovery_results_table(result.trending_stocks)
    console.print(results_table)
    console.print()

    while True:
        selected_stock = select_stock_for_detail(result.trending_stocks)

        if selected_stock is None:
            break

        rank = result.trending_stocks.index(selected_stock) + 1
        detail_panel = create_stock_detail_panel(selected_stock, rank)
        console.print()
        console.print(detail_panel)
        console.print()

        analyze_choice = questionary.confirm(
            f"Analyze {selected_stock.ticker}?",
            default=False,
            style=questionary.Style(
                [
                    ("selected", "fg:green noinherit"),
                    ("highlighted", "fg:green noinherit"),
                ]
            ),
        ).ask()

        if analyze_choice:
            console.print(f"\n[green]Starting analysis for {selected_stock.ticker}...[/green]\n")
            run_analysis_for_ticker(selected_stock.ticker, config)
            break


def run_analysis_for_ticker(ticker: str, config: dict):
    """
    Run an interactive, multi-agent analysis workflow for a single ticker and persist live reports and logs.
    
    This function launches an interactive CLI flow to select analyst agents, research depth, and deep-thinking model, then builds and executes a TradingAgentsGraph analysis for the given ticker. While the graph streams results it updates the global message buffer and a live terminal UI, writes per-section Markdown reports into a results directory, and appends structured messages and tool-call entries to a per-run log file.
    
    Parameters:
        ticker (str): The stock ticker symbol to analyze.
        config (dict): Configuration options that control the analysis and persistence. Recognized keys:
            - "results_dir" (str | Path): Base directory where run results, reports, and logs are written.
            - "llm_provider" (str): Name of the LLM provider used to select deep-thinking agents.
            - "max_debate_rounds" (int): (set by this function from selected research depth) maximum rounds for debate-style analysis.
            - "max_risk_discuss_rounds" (int): (set by this function from selected research depth) maximum rounds for risk discussion.
            - "deep_think_llm" (str): (set by this function) identifier of the deep-thinking LLM selected for the run.
    
    Side effects:
        - Interacts with the user via the terminal (selection prompts and live progress).
        - Creates directories and files under `<results_dir>/<ticker>/<YYYY-MM-DD>/`, including per-section Markdown reports and a message_tool.log file.
        - Updates the global `message_buffer` with streamed messages, tool calls, agent statuses, and report sections.
    """
    analysis_date = datetime.datetime.now().strftime("%Y-%m-%d")

    console.print(
        create_question_box(
            "Analysts Team",
            "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts()
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    console.print(
        create_question_box(
            "Research Depth",
            "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()

    console.print(
        create_question_box(
            "Deep-Thinking Model",
            "Select the model for deep analysis"
        )
    )
    llm_provider = config.get("llm_provider", "openai")
    selected_deep_thinker = select_deep_thinking_agent(llm_provider.capitalize())

    config["max_debate_rounds"] = selected_research_depth
    config["max_risk_discuss_rounds"] = selected_research_depth
    config["deep_think_llm"] = selected_deep_thinker

    graph = TradingAgentsGraph(
        [analyst.value for analyst in selected_analysts], config=config, debug=True
    )

    results_dir = Path(config["results_dir"]) / ticker / analysis_date
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        """
        Create a wrapper for a method that appends the object's most recent message to the log file after the method runs.
        
        Parameters:
            obj: The object containing a `messages` deque where the last entry is a (timestamp, message_type, content) tuple.
            func_name (str): The name of the method on `obj` to wrap.
        
        Returns:
            function: A wrapper function that calls the original method and then writes the latest message (with newlines replaced by spaces) to the module-level `log_file`.
        """
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Call the wrapped function and append the most recent message from obj.messages to the log file.
            
            Calls the original function with the provided arguments, then reads the last (timestamp, message_type, content) tuple from obj.messages, replaces newlines in the content with spaces, and writes a single-line entry "<timestamp> [<message_type>] <content>" to the configured log file.
            """
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper

    def save_tool_call_decorator(obj, func_name):
        """
        Create a wrapper for a method on `obj` that, after invoking the original method, appends the object's most recent tool-call entry to the shared log file.
        
        Parameters:
            obj: An object that exposes a `tool_calls` sequence where each entry is a tuple (timestamp, tool_name, args_dict).
            func_name (str): The name of the attribute on `obj` to wrap; that attribute must be callable.
        
        Returns:
            wrapper (callable): A function that calls the original method and then writes a single line to `log_file` containing the timestamp, the tool name, and the keyword-argument pairs from the recorded tool call.
        """
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Call the wrapped function with the supplied arguments and append the most recent tool call recorded on `obj.tool_calls` to `log_file`.
            
            This writes a single line to `log_file` with the format:
            `<timestamp> [Tool Call] <tool_name>(key1=value1, key2=value2, ...)`.
            
            Note: forwards all positional and keyword arguments to the wrapped function and relies on `obj.tool_calls` containing at least one entry.
            """
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        """
        Wraps an object's method to persist updated report sections to markdown files.
        
        Calls the original method named by `func_name` on `obj` with (section_name, content). After the call, if `obj.report_sections[section_name]` exists and is non-empty, writes that content to a file named "<section_name>.md" in the module's `report_dir`.
        
        Parameters:
            obj: The target object that exposes a callable attribute named by `func_name` and a `report_sections` mapping.
            func_name (str): The attribute name of the method on `obj` to wrap.
        
        Returns:
            function: A wrapper function that accepts `section_name` and `content`, performs the original call, and persists the section to disk when present.
        """
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            """
            Persist a report section to disk when the section exists and has content.
            
            Parameters:
            	section_name (str): Key identifying the report section (used as the output filename `<section_name>.md`).
            	content (str | None): Section content provided by the caller; if updated content is available on `obj.report_sections`, that value is written instead.
            
            Notes:
            	If the section exists in `obj.report_sections` and its value is not None or empty, the function writes the section content to `<report_dir>/<section_name>.md`.
            """
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    with open(report_dir / file_name, "w") as f:
                        f.write(content)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        update_display(layout)

        message_buffer.add_message("System", f"Selected ticker: {ticker}")
        message_buffer.add_message("System", f"Analysis date: {analysis_date}")
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selected_analysts)}",
        )
        update_display(layout)

        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "pending")

        for section in message_buffer.report_sections:
            message_buffer.report_sections[section] = None
        message_buffer.current_report = None
        message_buffer.final_report = None

        first_analyst = f"{selected_analysts[0].value.capitalize()} Analyst"
        message_buffer.update_agent_status(first_analyst, "in_progress")
        update_display(layout)

        spinner_text = f"Analyzing {ticker} on {analysis_date}..."
        update_display(layout, spinner_text)

        init_agent_state = graph.propagator.create_initial_state(ticker, analysis_date)
        args = graph.propagator.get_graph_args()

        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            if len(chunk["messages"]) > 0:
                last_message = chunk["messages"][-1]

                if hasattr(last_message, "content"):
                    content = extract_content_string(last_message.content)
                    msg_type = "Reasoning"
                else:
                    content = str(last_message)
                    msg_type = "System"

                message_buffer.add_message(msg_type, content)

                if hasattr(last_message, "tool_calls"):
                    for tool_call in last_message.tool_calls:
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

                process_chunk_for_display(chunk, selected_analysts)
                update_display(layout)

            trace.append(chunk)

        final_state = trace[-1]
        decision = graph.process_signal(final_state["final_trade_decision"])

        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message("Analysis", f"Completed analysis for {analysis_date}")

        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        display_complete_report(final_state)
        update_display(layout)


def process_chunk_for_display(chunk, selected_analysts):
    """
    Process a streaming graph chunk and update the global message buffer and agent statuses for UI display.
    
    This inspects known keys in `chunk` (market_report, sentiment_report, news_report, fundamentals_report,
    investment_debate_state, trader_investment_plan, risk_debate_state) and, when present, writes report sections,
    appends reasoning messages, and advances or completes the corresponding agent statuses to reflect pipeline progress.
    
    Parameters:
        chunk (dict): A streaming update containing zero or more report fragments and decision-state entries.
        selected_analysts (Iterable[AnalystType]): The set of analyst types enabled for this run; used to advance the next agent when a stage completes.
    
    """
    if "market_report" in chunk and chunk["market_report"]:
        message_buffer.update_report_section("market_report", chunk["market_report"])
        message_buffer.update_agent_status("Market Analyst", "completed")
        if AnalystType.SOCIAL in selected_analysts:
            message_buffer.update_agent_status("Social Analyst", "in_progress")

    if "sentiment_report" in chunk and chunk["sentiment_report"]:
        message_buffer.update_report_section("sentiment_report", chunk["sentiment_report"])
        message_buffer.update_agent_status("Social Analyst", "completed")
        if AnalystType.NEWS in selected_analysts:
            message_buffer.update_agent_status("News Analyst", "in_progress")

    if "news_report" in chunk and chunk["news_report"]:
        message_buffer.update_report_section("news_report", chunk["news_report"])
        message_buffer.update_agent_status("News Analyst", "completed")
        if AnalystType.FUNDAMENTALS in selected_analysts:
            message_buffer.update_agent_status("Fundamentals Analyst", "in_progress")

    if "fundamentals_report" in chunk and chunk["fundamentals_report"]:
        message_buffer.update_report_section("fundamentals_report", chunk["fundamentals_report"])
        message_buffer.update_agent_status("Fundamentals Analyst", "completed")
        update_research_team_status("in_progress")

    if "investment_debate_state" in chunk and chunk["investment_debate_state"]:
        debate_state = chunk["investment_debate_state"]

        if "bull_history" in debate_state and debate_state["bull_history"]:
            update_research_team_status("in_progress")
            bull_responses = debate_state["bull_history"].split("\n")
            latest_bull = bull_responses[-1] if bull_responses else ""
            if latest_bull:
                message_buffer.add_message("Reasoning", latest_bull)
                message_buffer.update_report_section(
                    "investment_plan",
                    f"### Bull Researcher Analysis\n{latest_bull}",
                )

        if "bear_history" in debate_state and debate_state["bear_history"]:
            update_research_team_status("in_progress")
            bear_responses = debate_state["bear_history"].split("\n")
            latest_bear = bear_responses[-1] if bear_responses else ""
            if latest_bear:
                message_buffer.add_message("Reasoning", latest_bear)
                message_buffer.update_report_section(
                    "investment_plan",
                    f"{message_buffer.report_sections['investment_plan']}\n\n### Bear Researcher Analysis\n{latest_bear}",
                )

        if "judge_decision" in debate_state and debate_state["judge_decision"]:
            update_research_team_status("in_progress")
            message_buffer.add_message(
                "Reasoning",
                f"Research Manager: {debate_state['judge_decision']}",
            )
            message_buffer.update_report_section(
                "investment_plan",
                f"{message_buffer.report_sections['investment_plan']}\n\n### Research Manager Decision\n{debate_state['judge_decision']}",
            )
            update_research_team_status("completed")
            message_buffer.update_agent_status("Risky Analyst", "in_progress")

    if "trader_investment_plan" in chunk and chunk["trader_investment_plan"]:
        message_buffer.update_report_section("trader_investment_plan", chunk["trader_investment_plan"])
        message_buffer.update_agent_status("Risky Analyst", "in_progress")

    if "risk_debate_state" in chunk and chunk["risk_debate_state"]:
        risk_state = chunk["risk_debate_state"]

        if "current_risky_response" in risk_state and risk_state["current_risky_response"]:
            message_buffer.update_agent_status("Risky Analyst", "in_progress")
            message_buffer.add_message(
                "Reasoning",
                f"Risky Analyst: {risk_state['current_risky_response']}",
            )
            message_buffer.update_report_section(
                "final_trade_decision",
                f"### Risky Analyst Analysis\n{risk_state['current_risky_response']}",
            )

        if "current_safe_response" in risk_state and risk_state["current_safe_response"]:
            message_buffer.update_agent_status("Safe Analyst", "in_progress")
            message_buffer.add_message(
                "Reasoning",
                f"Safe Analyst: {risk_state['current_safe_response']}",
            )
            message_buffer.update_report_section(
                "final_trade_decision",
                f"### Safe Analyst Analysis\n{risk_state['current_safe_response']}",
            )

        if "current_neutral_response" in risk_state and risk_state["current_neutral_response"]:
            message_buffer.update_agent_status("Neutral Analyst", "in_progress")
            message_buffer.add_message(
                "Reasoning",
                f"Neutral Analyst: {risk_state['current_neutral_response']}",
            )
            message_buffer.update_report_section(
                "final_trade_decision",
                f"### Neutral Analyst Analysis\n{risk_state['current_neutral_response']}",
            )

        if "judge_decision" in risk_state and risk_state["judge_decision"]:
            message_buffer.update_agent_status("Portfolio Manager", "in_progress")
            message_buffer.add_message(
                "Reasoning",
                f"Portfolio Manager: {risk_state['judge_decision']}",
            )
            message_buffer.update_report_section(
                "final_trade_decision",
                f"### Portfolio Manager Decision\n{risk_state['judge_decision']}",
            )
            message_buffer.update_agent_status("Risky Analyst", "completed")
            message_buffer.update_agent_status("Safe Analyst", "completed")
            message_buffer.update_agent_status("Neutral Analyst", "completed")
            message_buffer.update_agent_status("Portfolio Manager", "completed")


def create_layout():
    """
    Constructs the Rich UI layout used across the CLI with predefined header, main, and footer regions.
    
    The layout contains a top header, a main area split into an upper panel (progress and messages) and an analysis panel, and a bottom footer. The upper panel is further split into a progress column and a messages column.
    
    Returns:
        layout (rich.layout.Layout): A Layout with named regions: "header", "main", "footer", where "main" contains "upper" (with "progress" and "messages") and "analysis".
    """
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def update_display(layout, spinner_text=None):
    """
    Update the given Rich layout panels to reflect the current CLI state, including header, per-team progress, recent messages/tool calls, the current analysis report, and footer statistics.
    
    Parameters:
        layout (rich.layout.Layout): The layout to update; panels named "header", "progress", "messages", "analysis", and "footer" will be modified.
        spinner_text (str | None): Optional text to append as a spinner row in the messages panel when provided.
    
    Behavior notes:
        - Displays per-team agent statuses and shows a spinner for agents with status "in_progress".
        - Shows the most recent 12 combined tool calls and messages (long tool args are truncated at 100 chars; message content is truncated at 200 chars).
        - Sorts messages chronologically and renders wrapped content for display.
        - When a current report is available, renders it in the analysis panel; otherwise displays a waiting placeholder.
        - Footer presents counts for tool calls, LLM reasoning calls, and generated report sections.
    """
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]Built by Tauric Research (https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,
        title=None,
        padding=(0, 2),
        expand=True,
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    teams = {
        "Analyst Team": [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Risky Analyst", "Neutral Analyst", "Safe Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    for team, agents in teams.items():
        first_agent = agents[0]
        status = message_buffer.agent_status[first_agent]
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        for agent in agents[1:]:
            status = message_buffer.agent_status[agent]
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        progress_table.add_row("-" * 20, "-" * 20, "-" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,
        box=box.MINIMAL,
        show_lines=True,
        padding=(0, 1),
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column("Content", style="white", no_wrap=False, ratio=1)

    all_messages = []

    for timestamp, tool_name, args in message_buffer.tool_calls:
        if isinstance(args, str) and len(args) > 100:
            args = args[:97] + "..."
        all_messages.append((timestamp, "Tool", f"{tool_name}: {args}"))

    for timestamp, msg_type, content in message_buffer.messages:
        content_str = content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'tool_use':
                        text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
                else:
                    text_parts.append(str(item))
            content_str = ' '.join(text_parts)
        elif not isinstance(content_str, str):
            content_str = str(content)

        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    all_messages.sort(key=lambda x: x[0])
    max_messages = 12
    recent_messages = all_messages[-max_messages:]

    for timestamp, msg_type, content in recent_messages:
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    if spinner_text:
        messages_table.add_row("", "Spinner", spinner_text)

    if len(all_messages) > max_messages:
        messages_table.footer = (
            f"[dim]Showing last {max_messages} of {len(all_messages)} messages[/dim]"
        )

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    tool_calls_count = len(message_buffer.tool_calls)
    llm_calls_count = sum(
        1 for _, msg_type, _ in message_buffer.messages if msg_type == "Reasoning"
    )
    reports_count = sum(
        1 for content in message_buffer.report_sections.values() if content is not None
    )

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(
        f"Tool Calls: {tool_calls_count} | LLM Calls: {llm_calls_count} | Generated Reports: {reports_count}"
    )

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """
    Collect interactive CLI inputs for a trading analysis run and return the gathered selections.
    
    Returns:
        selections (dict): Mapping of user choices:
            - "ticker" (str): Uppercase ticker symbol to analyze.
            - "analysis_date" (str): Analysis date in "YYYY-MM-DD" format.
            - "analysts" (list): Selected analyst agent enum values.
            - "research_depth" (str or enum): Chosen research depth level.
            - "llm_provider" (str): Selected LLM provider name (lowercased).
            - "backend_url" (str or None): Backend URL associated with the chosen provider.
            - "shallow_thinker" (str or enum): Selected shallow-thinking agent for analysis.
            - "deep_thinker" (str or enum): Selected deep-thinking agent for analysis.
    """
    with open("./cli/static/welcome.txt", "r") as f:
        welcome_ascii = f.read()

    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team -> II. Research Team -> III. Trader -> IV. Risk Management -> V. Portfolio Management\n\n"
    welcome_content += "[dim]Built by Tauric Research (https://github.com/TauricResearch)[/dim]"

    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()

    console.print(
        create_question_box(
            "Step 1: Ticker Symbol", "Enter the ticker symbol to analyze", "SPY"
        )
    )
    selected_ticker = get_ticker()

    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    console.print(
        create_question_box(
            "Step 3: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts()
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    console.print(
        create_question_box(
            "Step 4: Research Depth", "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()

    console.print(
        create_question_box(
            "Step 5: OpenAI backend", "Select which service to talk to"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()

    console.print(
        create_question_box(
            "Step 6: Thinking Agents", "Select your thinking agents for analysis"
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    return {
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
    }


def get_ticker():
    """
    Prompt the user to enter a stock ticker symbol, defaulting to "SPY".
    
    Returns:
        ticker (str): The ticker symbol entered by the user (or "SPY" if left blank).
    """
    return typer.prompt("", default="SPY")


def get_analysis_date():
    """
    Prompt the user to select an analysis date (YYYY-MM-DD), defaulting to today.
    
    Prompts repeatedly until the input matches the YYYY-MM-DD format and is not a future date.
    
    Returns:
        date_str (str): The chosen analysis date in 'YYYY-MM-DD' format.
    """
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def display_complete_report(final_state):
    """
    Render a multi-section, human-readable analysis report to the application console from a final analysis state.
    
    Parameters:
        final_state (dict): A mapping containing completed report sections and decisions. Known keys:
            - "market_report" (str): Market analyst markdown content.
            - "sentiment_report" (str): Social/sentiment analyst markdown content.
            - "news_report" (str): News analyst markdown content.
            - "fundamentals_report" (str): Fundamentals analyst markdown content.
            - "investment_debate_state" (dict): Research-team debate with optional keys:
                - "bull_history" (str): Bull researcher markdown content.
                - "bear_history" (str): Bear researcher markdown content.
                - "judge_decision" (str): Research manager decision markdown content.
            - "trader_investment_plan" (str): Trader team plan markdown content.
            - "risk_debate_state" (dict): Risk/portfolio debate with optional keys:
                - "risky_history" (str): Aggressive analyst markdown content.
                - "safe_history" (str): Conservative analyst markdown content.
                - "neutral_history" (str): Neutral analyst markdown content.
                - "judge_decision" (str): Portfolio manager decision markdown content.
    
    Behavior:
        Prints formatted panels and columns to the global console for each present section in the order:
        Analyst Team Reports, Research Team Decision, Trading Team Plan, Risk Management Team Decision, and Portfolio Manager Decision.
    """
    console.print("\n[bold green]Complete Analysis Report[/bold green]\n")

    analyst_reports = []

    if final_state.get("market_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["market_report"]),
                title="Market Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if final_state.get("sentiment_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["sentiment_report"]),
                title="Social Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if final_state.get("news_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["news_report"]),
                title="News Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if final_state.get("fundamentals_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["fundamentals_report"]),
                title="Fundamentals Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if analyst_reports:
        console.print(
            Panel(
                Columns(analyst_reports, equal=True, expand=True),
                title="I. Analyst Team Reports",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    if final_state.get("investment_debate_state"):
        research_reports = []
        debate_state = final_state["investment_debate_state"]

        if debate_state.get("bull_history"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["bull_history"]),
                    title="Bull Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if debate_state.get("bear_history"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["bear_history"]),
                    title="Bear Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if debate_state.get("judge_decision"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["judge_decision"]),
                    title="Research Manager",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if research_reports:
            console.print(
                Panel(
                    Columns(research_reports, equal=True, expand=True),
                    title="II. Research Team Decision",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )

    if final_state.get("trader_investment_plan"):
        console.print(
            Panel(
                Panel(
                    Markdown(final_state["trader_investment_plan"]),
                    title="Trader",
                    border_style="blue",
                    padding=(1, 2),
                ),
                title="III. Trading Team Plan",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    if final_state.get("risk_debate_state"):
        risk_reports = []
        risk_state = final_state["risk_debate_state"]

        if risk_state.get("risky_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["risky_history"]),
                    title="Aggressive Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if risk_state.get("safe_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["safe_history"]),
                    title="Conservative Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if risk_state.get("neutral_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["neutral_history"]),
                    title="Neutral Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if risk_reports:
            console.print(
                Panel(
                    Columns(risk_reports, equal=True, expand=True),
                    title="IV. Risk Management Team Decision",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        if risk_state.get("judge_decision"):
            console.print(
                Panel(
                    Panel(
                        Markdown(risk_state["judge_decision"]),
                        title="Portfolio Manager",
                        border_style="blue",
                        padding=(1, 2),
                    ),
                    title="V. Portfolio Manager Decision",
                    border_style="green",
                    padding=(1, 2),
                )
            )


def update_research_team_status(status):
    """
    Set the status for all members of the research team in the global message buffer.
    
    Parameters:
        status (str): The status value to assign to each research team agent (e.g., "pending", "in_progress", "completed").
    """
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager", "Trader"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)


def extract_content_string(content):
    """
    Convert various content forms into a single plain string suitable for display.
    
    Parameters:
        content: The value to stringify. May be a string, a list of items (where items can be dicts or other values), or any other object.
    
    Returns:
        A single string representing the input. If `content` is a string it is returned unchanged. If it is a list, elements are concatenated with spaces; dict items with `type == "text"` contribute their `text` value, dict items with `type == "tool_use"` are represented as `[Tool: <name>]`, and other elements are converted with `str()`. For other types, the result of `str(content)` is returned.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif item.get('type') == 'tool_use':
                    text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
            else:
                text_parts.append(str(item))
        return ' '.join(text_parts)
    else:
        return str(content)


def run_analysis():
    """
    Run the interactive analysis workflow for a single ticker.
    
    Starts an interactive prompt sequence to collect analysis options, constructs and runs a TradingAgentsGraph analysis, streams agent messages and report sections to the live CLI display, and persists logs and per-section Markdown reports to disk.
    
    Side effects:
    - Prompts the user for selections (ticker, date, analysts, models, backend).
    - Creates results and report directories and writes a per-run log file and section Markdown files.
    - Updates the global message_buffer with messages, tool calls, agent statuses, and report sections.
    - Renders a live Rich UI during streaming and displays the final consolidated report.
    """
    selections = get_user_selections()

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()

    graph = TradingAgentsGraph(
        [analyst.value for analyst in selections["analysts"]], config=config, debug=True
    )

    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        """
        Create a wrapper for a method that appends the object's most recent message to the log file after the method runs.
        
        Parameters:
            obj: The object containing a `messages` deque where the last entry is a (timestamp, message_type, content) tuple.
            func_name (str): The name of the method on `obj` to wrap.
        
        Returns:
            function: A wrapper function that calls the original method and then writes the latest message (with newlines replaced by spaces) to the module-level `log_file`.
        """
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Call the wrapped function and append the most recent message from obj.messages to the log file.
            
            Calls the original function with the provided arguments, then reads the last (timestamp, message_type, content) tuple from obj.messages, replaces newlines in the content with spaces, and writes a single-line entry "<timestamp> [<message_type>] <content>" to the configured log file.
            """
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper

    def save_tool_call_decorator(obj, func_name):
        """
        Create a wrapper for a method on `obj` that, after invoking the original method, appends the object's most recent tool-call entry to the shared log file.
        
        Parameters:
            obj: An object that exposes a `tool_calls` sequence where each entry is a tuple (timestamp, tool_name, args_dict).
            func_name (str): The name of the attribute on `obj` to wrap; that attribute must be callable.
        
        Returns:
            wrapper (callable): A function that calls the original method and then writes a single line to `log_file` containing the timestamp, the tool name, and the keyword-argument pairs from the recorded tool call.
        """
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    with open(report_dir / file_name, "w") as f:
                        f.write(content)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        update_display(layout)

        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout)

        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "pending")

        for section in message_buffer.report_sections:
            message_buffer.report_sections[section] = None
        message_buffer.current_report = None
        message_buffer.final_report = None

        first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
        message_buffer.update_agent_status(first_analyst, "in_progress")
        update_display(layout)

        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text)

        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"]
        )
        args = graph.propagator.get_graph_args()

        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            if len(chunk["messages"]) > 0:
                last_message = chunk["messages"][-1]

                if hasattr(last_message, "content"):
                    content = extract_content_string(last_message.content)
                    msg_type = "Reasoning"
                else:
                    content = str(last_message)
                    msg_type = "System"

                message_buffer.add_message(msg_type, content)

                if hasattr(last_message, "tool_calls"):
                    for tool_call in last_message.tool_calls:
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(
                                tool_call["name"], tool_call["args"]
                            )
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

                if "market_report" in chunk and chunk["market_report"]:
                    message_buffer.update_report_section(
                        "market_report", chunk["market_report"]
                    )
                    message_buffer.update_agent_status("Market Analyst", "completed")
                    if "social" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Social Analyst", "in_progress"
                        )

                if "sentiment_report" in chunk and chunk["sentiment_report"]:
                    message_buffer.update_report_section(
                        "sentiment_report", chunk["sentiment_report"]
                    )
                    message_buffer.update_agent_status("Social Analyst", "completed")
                    if "news" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "News Analyst", "in_progress"
                        )

                if "news_report" in chunk and chunk["news_report"]:
                    message_buffer.update_report_section(
                        "news_report", chunk["news_report"]
                    )
                    message_buffer.update_agent_status("News Analyst", "completed")
                    if "fundamentals" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Fundamentals Analyst", "in_progress"
                        )

                if "fundamentals_report" in chunk and chunk["fundamentals_report"]:
                    message_buffer.update_report_section(
                        "fundamentals_report", chunk["fundamentals_report"]
                    )
                    message_buffer.update_agent_status(
                        "Fundamentals Analyst", "completed"
                    )
                    update_research_team_status("in_progress")

                if (
                    "investment_debate_state" in chunk
                    and chunk["investment_debate_state"]
                ):
                    debate_state = chunk["investment_debate_state"]

                    if "bull_history" in debate_state and debate_state["bull_history"]:
                        update_research_team_status("in_progress")
                        bull_responses = debate_state["bull_history"].split("\n")
                        latest_bull = bull_responses[-1] if bull_responses else ""
                        if latest_bull:
                            message_buffer.add_message("Reasoning", latest_bull)
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"### Bull Researcher Analysis\n{latest_bull}",
                            )

                    if "bear_history" in debate_state and debate_state["bear_history"]:
                        update_research_team_status("in_progress")
                        bear_responses = debate_state["bear_history"].split("\n")
                        latest_bear = bear_responses[-1] if bear_responses else ""
                        if latest_bear:
                            message_buffer.add_message("Reasoning", latest_bear)
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"{message_buffer.report_sections['investment_plan']}\n\n### Bear Researcher Analysis\n{latest_bear}",
                            )

                    if (
                        "judge_decision" in debate_state
                        and debate_state["judge_decision"]
                    ):
                        update_research_team_status("in_progress")
                        message_buffer.add_message(
                            "Reasoning",
                            f"Research Manager: {debate_state['judge_decision']}",
                        )
                        message_buffer.update_report_section(
                            "investment_plan",
                            f"{message_buffer.report_sections['investment_plan']}\n\n### Research Manager Decision\n{debate_state['judge_decision']}",
                        )
                        update_research_team_status("completed")
                        message_buffer.update_agent_status(
                            "Risky Analyst", "in_progress"
                        )

                if (
                    "trader_investment_plan" in chunk
                    and chunk["trader_investment_plan"]
                ):
                    message_buffer.update_report_section(
                        "trader_investment_plan", chunk["trader_investment_plan"]
                    )
                    message_buffer.update_agent_status("Risky Analyst", "in_progress")

                if "risk_debate_state" in chunk and chunk["risk_debate_state"]:
                    risk_state = chunk["risk_debate_state"]

                    if (
                        "current_risky_response" in risk_state
                        and risk_state["current_risky_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Risky Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Risky Analyst: {risk_state['current_risky_response']}",
                        )
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Risky Analyst Analysis\n{risk_state['current_risky_response']}",
                        )

                    if (
                        "current_safe_response" in risk_state
                        and risk_state["current_safe_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Safe Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Safe Analyst: {risk_state['current_safe_response']}",
                        )
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Safe Analyst Analysis\n{risk_state['current_safe_response']}",
                        )

                    if (
                        "current_neutral_response" in risk_state
                        and risk_state["current_neutral_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Neutral Analyst: {risk_state['current_neutral_response']}",
                        )
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Neutral Analyst Analysis\n{risk_state['current_neutral_response']}",
                        )

                    if "judge_decision" in risk_state and risk_state["judge_decision"]:
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Portfolio Manager: {risk_state['judge_decision']}",
                        )
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Portfolio Manager Decision\n{risk_state['judge_decision']}",
                        )
                        message_buffer.update_agent_status("Risky Analyst", "completed")
                        message_buffer.update_agent_status("Safe Analyst", "completed")
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "completed"
                        )
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "completed"
                        )

                update_display(layout)

            trace.append(chunk)

        final_state = trace[-1]
        decision = graph.process_signal(final_state["final_trade_decision"])

        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "Analysis", f"Completed analysis for {selections['analysis_date']}"
        )

        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        display_complete_report(final_state)

        update_display(layout)


def show_main_menu():
    """
    Display the main menu UI and prompt the user to choose an action.
    
    Reads and renders the welcome screen, then presents choices for analyzing a specific stock or discovering trending stocks. Exits the process if the user cancels or makes no selection.
    
    Returns:
        choice (str): 'analyze' when the user selects "Analyze a specific stock", 'discover' when the user selects "Discover trending stocks".
    """
    with open("./cli/static/welcome.txt", "r") as f:
        welcome_ascii = f.read()

    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Available Options:[/bold]\n"
    welcome_content += "1. Analyze a specific stock\n"
    welcome_content += "2. Discover trending stocks\n\n"
    welcome_content += "[dim]Built by Tauric Research (https://github.com/TauricResearch)[/dim]"

    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()

    MENU_OPTIONS = [
        ("1. Analyze a specific stock", "analyze"),
        ("2. Discover trending stocks", "discover"),
    ]

    choice = questionary.select(
        "Select an option:",
        choices=[
            questionary.Choice(display, value=value) for display, value in MENU_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:green noinherit"),
                ("highlighted", "fg:green noinherit"),
                ("pointer", "fg:green noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No option selected. Exiting...[/red]")
        exit(0)

    return choice


@app.command()
def analyze():
    """
    Launch the interactive stock analysis command that runs the full analysis workflow.
    """
    run_analysis()


@app.command()
def discover():
    """
    Start the interactive trending-stocks discovery workflow.
    
    Prompts the user for lookback period, optional sector/event filters, and model/backend choices, runs the discovery process, displays progress and results, and saves discovery results when completed.
    """
    discover_trending_flow()


@app.command()
def menu():
    """
    Present the main menu and route the user to the selected workflow.
    
    Displays the top-level menu, then starts the Analyze workflow when the user chooses "analyze" or the Discover trending-stocks workflow when the user chooses "discover".
    """
    choice = show_main_menu()
    if choice == "analyze":
        run_analysis()
    elif choice == "discover":
        discover_trending_flow()


if __name__ == "__main__":
    choice = show_main_menu()
    if choice == "analyze":
        run_analysis()
    elif choice == "discover":
        discover_trending_flow()