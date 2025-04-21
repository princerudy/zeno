# server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

@mcp.tool()
def order_pizza(
    size: str = "medium",
    toppings: list = ["pepperoni", "mushrooms"],
    extra_cheese: bool = False,
) -> str:
    """Order a pizza with the specified size and toppings"""
    order_details = f"Ordering a {size} pizza with {', '.join(toppings)}"
    if extra_cheese:
        order_details += " and extra cheese"
    return order_details