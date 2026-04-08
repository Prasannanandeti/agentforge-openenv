from typing import Dict, List, Any, Optional

# -------------------------

# Mock Database

# -------------------------

ORDER_DB = {
"ORD-101": {
"status": "Shipped",
"delivery_date": "2023-10-25",
"items": ["Laptop"],
"customer": "Alice"
},
"ORD-202": {
"status": "Delivered",
"delivery_date": "2023-10-20",
"items": ["Headphones"],
"customer": "Bob",
"refund_eligible": True
},
"ORD-303": {
"status": "Processing",
"items": ["Monitor"],
"customer": "Charlie",
"refund_eligible": False
}
}

# -------------------------

# Expected Agent Flows

# -------------------------

EXPECTED_FLOWS = {
"easy_1": ["call_tool", "reply", "close_ticket"],
"medium_1": ["call_tool", "call_tool", "reply", "close_ticket"],
"hard_1": ["ask_info", "call_tool", "reply", "close_ticket"]
}

# -------------------------

# Empathy Keywords

# -------------------------

EMPATHY_KEYWORDS = ["sorry", "apologize", "understand", "regret", "patience"]

# -------------------------

# Tasks

# -------------------------

TASKS = [
{
"id": "easy_1",
"name": "Order Status Check",
"difficulty": "EASY",
"initial_query": "Hi, I'm Alice. Where is my order ORD-101?",
"context": {"user_id": "Alice", "order_id": "ORD-101"},
"available_tools": ["get_order_details"]
},
{
"id": "medium_1",
"name": "Refund Processing",
"difficulty": "MEDIUM",
"initial_query": "I want a refund for my headphones (ORD-202).",
"context": {"user_id": "Bob", "order_id": "ORD-202"},
"available_tools": ["get_order_details", "process_refund"]
},
{
"id": "hard_1",
"name": "Angry Customer Missing Info",
"difficulty": "HARD",
"initial_query": "Your service is terrible! I want my money back for the broken monitor I bought last week!",
"context": {"user_id": "Charlie", "order_id": None},  # Must ask for order_id
"available_tools": ["get_order_details", "process_refund"]
}
]

# -------------------------

# Tool Simulation

# -------------------------

def mock_tool_call(name: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not params or "order_id" not in params or not params["order_id"]:
        return {
            "error": "Missing mandatory parameter: order_id",
            "status": "failed"
        }

    oid = params.get("order_id")

    if name == "get_order_details":
        data = ORDER_DB.get(oid)
        return {
            "status": "success",
            "data": data if data else "Order not found"
        }

    elif name == "process_refund":
        order = ORDER_DB.get(oid)

        if order and order.get("refund_eligible"):
            return {
                "status": "success",
                "message": f"Refund successful for {oid}"
            }

        return {
            "status": "denied",
            "message": f"Refund rejected: Policy violation for {oid}"
        }

    return {
        "error": "Unknown tool",
        "status": "failed"
    }

