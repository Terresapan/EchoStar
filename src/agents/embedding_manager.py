"""
EmbeddingManager for optimizing memory retrieval operations.

This module provides caching and batching functionality for embedding operations
to reduce API calls and improve performance in memory retrieval.
"""

import hashlib
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
from langchain_core.tools import BaseTool

# Import logging utilities
try:
    from ..utils.logging_utils import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback for testing or standalone usage
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Mock performance_timer for testing
    class MockPerformanceTimer:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    logger.performance_timer = MockPerformanceTimer


class EmbeddingManager:
    """
    Manages embedding operations with caching and batching to optimize API usage.
    
    Features:
    - LRU cache for embedding results to avoid duplicate API calls
    - Batch processing for multiple embedding requests
    - Cache hit/miss tracking and metrics
    - Automatic cache cleanup and management
    """
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        """
        Initialize the EmbeddingManager.
        
        Args:
            cache_size: Maximum number of embeddings to cache
            cache_ttl: Time-to-live for cache entries in seconds
        """
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.embedding_cache: Dict[str, Tuple[List[float], float]] = {}
        self.search_cache: Dict[str, Tuple[List[Any], float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls_saved = 0
        
        logger.info(f"EmbeddingManager initialized with cache_size={cache_size}, cache_ttl={cache_ttl}")
    
    def _generate_cache_key(self, text: str, search_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a unique cache key for the given text and search parameters.
        
        Args:
            text: The text to generate a key for
            search_params: Additional search parameters to include in the key
            
        Returns:
            A unique cache key string
        """
        # Create a hash of the text and parameters for consistent caching
        content = text
        if search_params:
            content += json.dumps(search_params, sort_keys=True)
        
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """
        Check if a cache entry is still valid based on TTL.
        
        Args:
            timestamp: The timestamp when the entry was cached
            
        Returns:
            True if the cache entry is still valid, False otherwise
        """
        return time.time() - timestamp < self.cache_ttl
    
    def _cleanup_expired_cache(self):
        """Remove expired entries from both caches."""
        current_time = time.time()
        
        # Clean embedding cache
        expired_embedding_keys = [
            key for key, (_, timestamp) in self.embedding_cache.items()
            if current_time - timestamp >= self.cache_ttl
        ]
        for key in expired_embedding_keys:
            del self.embedding_cache[key]
        
        # Clean search cache
        expired_search_keys = [
            key for key, (_, timestamp) in self.search_cache.items()
            if current_time - timestamp >= self.cache_ttl
        ]
        for key in expired_search_keys:
            del self.search_cache[key]
        
        if expired_embedding_keys or expired_search_keys:
            logger.debug(f"Cleaned up expired cache entries: embedding_expired={len(expired_embedding_keys)}, search_expired={len(expired_search_keys)}")
    
    def get_cached_search_result(self, query: str, search_params: Optional[Dict[str, Any]] = None) -> Optional[List[Any]]:
        """
        Get cached search results for a query.
        
        Args:
            query: The search query
            search_params: Additional search parameters
            
        Returns:
            Cached search results if available and valid, None otherwise
        """
        cache_key = self._generate_cache_key(query, search_params)
        
        if cache_key in self.search_cache:
            result, timestamp = self.search_cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.cache_hits += 1
                logger.debug(f"Search cache hit for query_length={len(query)}, cache_key={cache_key[:16]}...")
                return result
            else:
                # Remove expired entry
                del self.search_cache[cache_key]
        
        self.cache_misses += 1
        return None
    
    def cache_search_result(self, query: str, result: List[Any], search_params: Optional[Dict[str, Any]] = None):
        """
        Cache search results for a query.
        
        Args:
            query: The search query
            result: The search results to cache
            search_params: Additional search parameters
        """
        cache_key = self._generate_cache_key(query, search_params)
        current_time = time.time()
        
        # Implement LRU behavior by removing oldest entries if cache is full
        if len(self.search_cache) >= self.cache_size:
            # Remove the oldest entry
            oldest_key = min(self.search_cache.keys(), 
                           key=lambda k: self.search_cache[k][1])
            del self.search_cache[oldest_key]
        
        self.search_cache[cache_key] = (result, current_time)
        logger.debug(f"Cached search result: query_length={len(query)}, result_count={len(result) if result else 0}, cache_key={cache_key[:16]}...")
    
    def optimized_search(self, search_tool: BaseTool, query: str, **kwargs) -> List[Any]:
        """
        Perform an optimized search using the provided search tool with caching.
        
        Args:
            search_tool: The search tool to use
            query: The search query
            **kwargs: Additional parameters for the search tool
            
        Returns:
            Search results (from cache or fresh API call)
        """
        # Clean up expired cache entries periodically
        if len(self.search_cache) > 0 and len(self.search_cache) % 100 == 0:
            self._cleanup_expired_cache()
        
        # Check cache first
        search_params = kwargs if kwargs else None
        cached_result = self.get_cached_search_result(query, search_params)
        
        if cached_result is not None:
            self.api_calls_saved += 1
            logger.debug(f"Using cached search result: query_length={len(query)}, result_count={len(cached_result)}")
            return cached_result
        
        # Perform the actual search
        tool_name = search_tool.name if hasattr(search_tool, 'name') else 'unknown'
        logger.debug(f"Performing fresh search: query_length={len(query)}, tool_name={tool_name}")
        
        try:
            with logger.performance_timer("embedding_search", cached=False):
                result = search_tool.invoke(query)
            
            # Cache the result
            self.cache_search_result(query, result, search_params)
            
            logger.debug(f"Fresh search completed and cached: query_length={len(query)}, result_count={len(result) if result else 0}")
            
            return result
            
        except Exception as e:
            tool_name = search_tool.name if hasattr(search_tool, 'name') else 'unknown'
            logger.error(f"Search operation failed: error={str(e)}, query_length={len(query)}, tool_name={tool_name}")
            return []
    
    def batch_search(self, search_tool: BaseTool, queries: List[str], **kwargs) -> List[List[Any]]:
        """
        Perform batch searches with deduplication and caching.
        
        Args:
            search_tool: The search tool to use
            queries: List of search queries
            **kwargs: Additional parameters for the search tool
            
        Returns:
            List of search results for each query
        """
        if not queries:
            return []
        
        logger.info(f"Starting batch search: query_count={len(queries)}, unique_queries={len(set(queries))}")
        
        results = []
        unique_queries = {}  # Map query to result index
        
        # Deduplicate queries and check cache
        for i, query in enumerate(queries):
            if query in unique_queries:
                # Duplicate query - we'll reuse the result, count as API call saved
                self.api_calls_saved += 1
                continue
            
            # Check cache for this query
            cached_result = self.get_cached_search_result(query, kwargs if kwargs else None)
            if cached_result is not None:
                unique_queries[query] = cached_result
                # Note: api_calls_saved is already incremented in get_cached_search_result
            else:
                unique_queries[query] = None  # Mark for fresh search
        
        # Perform fresh searches for uncached queries
        fresh_searches = [query for query, result in unique_queries.items() if result is None]
        
        if fresh_searches:
            logger.debug(f"Performing fresh searches: fresh_count={len(fresh_searches)}, total_queries={len(queries)}")
            
            for query in fresh_searches:
                try:
                    with logger.performance_timer("batch_embedding_search", cached=False):
                        result = search_tool.invoke(query)
                    
                    unique_queries[query] = result
                    self.cache_search_result(query, result, kwargs if kwargs else None)
                    
                except Exception as e:
                    logger.error(f"Batch search failed for query: error={str(e)}, query_length={len(query)}")
                    unique_queries[query] = []
        
        # Build final results list maintaining original order
        for query in queries:
            results.append(unique_queries[query])
        
        logger.info(f"Batch search completed: total_queries={len(queries)}, fresh_searches={len(fresh_searches)}, api_calls_saved={len(queries) - len(fresh_searches)}")
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "api_calls_saved": self.api_calls_saved,
            "search_cache_size": len(self.search_cache),
            "embedding_cache_size": len(self.embedding_cache),
            "total_cache_size": len(self.search_cache) + len(self.embedding_cache)
        }
    
    def clear_cache(self):
        """Clear all cached data and reset statistics."""
        self.embedding_cache.clear()
        self.search_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls_saved = 0
        
        logger.info("All caches cleared and statistics reset")
    
    def clear_expired_cache(self):
        """Manually trigger cleanup of expired cache entries."""
        self._cleanup_expired_cache()