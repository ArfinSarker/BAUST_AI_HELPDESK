"""
Compatibility wrapper for retrieval operations.
Delegates to rag_search.py to maintain a single source of truth.
"""
from retrieval.rag_search import (
    reload_vector_store as reload_vector_store,
    search_information as search_information,
    search_with_metadata as search_with_metadata,
)

__all__ = ["search_information", "search_with_metadata", "reload_vector_store"]