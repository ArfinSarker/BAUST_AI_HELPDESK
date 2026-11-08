"""
Compatibility wrapper for retrieval operations.
Delegates to retrieval.rag_search to maintain a single source of truth.
"""
import warnings

# Warn developers to use the primary retrieval module path
warnings.warn(
    "Importing from 'retrieval.search' is deprecated. Please import directly from 'retrieval.rag_search'.",
    DeprecationWarning,
    stacklevel=2
)

try:
    from retrieval.rag_search import (
        reload_vector_store,
        search_information,
    )
    # Backward compatibility alias in case legacy code expects search_with_metadata
    search_with_metadata = search_information

except ImportError as e:
    print(f"[Compatibility Import Warning]: {e}")
    search_information = None
    search_with_metadata = None
    reload_vector_store = None

__all__ = ["search_information", "search_with_metadata", "reload_vector_store"]
