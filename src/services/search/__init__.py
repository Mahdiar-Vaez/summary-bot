"""Web search utilities for live external information retrieval."""

from src.services.search.web_search import (
    WebSearchService,
    extract_auto_search_query,
    should_presearch_query,
    strip_search_directives,
)

__all__ = [
    "WebSearchService",
    "extract_auto_search_query",
    "should_presearch_query",
    "strip_search_directives",
]
