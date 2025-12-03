from langchain_core.messages import HumanMessage, RemoveMessage

from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_sentiment,
    get_insider_transactions,
    get_global_news
)

def create_msg_delete():
    """
    Clear all messages and replace them with removal operations followed by a 'Continue' placeholder for Anthropic compatibility.
    
    Parameters:
        state (dict): Mapping that must include a "messages" iterable of message objects; each message object is expected to have an `id` attribute.
    
    Returns:
        dict: A mapping with key "messages" whose value is a list of `RemoveMessage` instances (one per original message) followed by a single `HumanMessage` with content "Continue".
    """
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        placeholder = HumanMessage(content="Continue")
        return {"messages": removal_operations + [placeholder]}
    return delete_messages