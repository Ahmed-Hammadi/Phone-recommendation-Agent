"""
MCP Tools Package
Contains specialized tools for phone recommendation system.
Each tool follows MCP protocol with proper schemas and validation.
"""

from .phone_specs import phone_specs_tool
from .web_scraper import web_scraper_tool
from .specs_analyzer import specs_analyzer_tool
from .sentiment_analyzer import sentiment_analyzer_tool
from .price_extractor import price_extractor_tool
from .alternative_recommender import alternative_recommender_tool
from .spec_validator import spec_validator_tool

# Tool registry for MCP server
TOOLS_REGISTRY = {
    "phone_specs": phone_specs_tool,
    "web_scraper": web_scraper_tool,
    "specs_analyzer": specs_analyzer_tool,
    "sentiment_analyzer": sentiment_analyzer_tool,
    "price_extractor": price_extractor_tool,
    "alternative_recommender": alternative_recommender_tool,
    "spec_validator": spec_validator_tool,
}

__all__ = [
    "phone_specs_tool",
    "web_scraper_tool",
    "specs_analyzer_tool",
    "sentiment_analyzer_tool", 
    "price_extractor_tool",
    "alternative_recommender_tool",
    "spec_validator_tool",
    "TOOLS_REGISTRY",
]
