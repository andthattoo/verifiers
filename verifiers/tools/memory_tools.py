"""
Memory tools for reading and writing to a graph-based memory buffer.
Implements a graph structure where nodes have unique IDs and metadata.
Provides search capability using BM25 for initial retrieval followed by
semantic search with rank fusion.
"""
from typing import Dict, List, Optional, Any, Union
from verifiers.utils.memory_layer import MemoryGraph

# Global memory graph instance
_memory_graph = MemoryGraph()


def write(metadata: Union[str, Dict[str, Any]], 
          connections: Optional[List[str]] = None) -> str:
    """
    Write a new node to memory with metadata and optional connections.
    
    Args:
        metadata: Either a string (treated as content) or a dictionary of metadata
        connections: Optional list of target node IDs to connect to
    
    Returns:
        The ID of the newly created node
    """
    node_id = _memory_graph.add_node(metadata)
    
    if connections:
        for target_id in connections:
            if target_id:
                _memory_graph.add_edge(node_id, target_id)
    
    return node_id


def search(query: str, top_k: int = 5, bfs_depth: int = 1, bm25_candidates: int = 10) -> List[Dict]:
    """
    Search memory using a two-stage process:
    1. BM25 retrieval + BFS exploration for initial candidates
    2. Semantic search with rank fusion for final results
    
    Args:
        query: The search query
        top_k: Number of results to return
        bfs_depth: Maximum depth for BFS exploration
        bm25_candidates: Number of initial candidates from BM25
    
    Returns:
        List of dicts containing node information
    """
    # If graph is empty, return empty results
    if not _memory_graph.nodes:
        return []
    
    # Step 1: BM25 retrieval for initial candidates
    initial_candidates = _memory_graph.bm25_search(query, bm25_candidates)
    
    # Step 2: BFS exploration from initial candidates
    candidate_set = _memory_graph.bfs_exploration(initial_candidates, bfs_depth)
    
    # Step 3: Semantic search on the candidates
    semantic_results = _memory_graph.semantic_search(query, list(candidate_set), top_k)
    
    # Step 4: Prepare results
    results = []
    for node_id, score in semantic_results:
        node = _memory_graph.get_node(node_id)
        if node:
            results.append({
                'id': node_id,
                'metadata': node.metadata,
                'score': score,
                'connections': list(node.edges)
            })
    
    return results


def update(node_id: str, metadata: Optional[Union[str, Dict[str, Any]]] = None,
           add_connections: Optional[List[str]] = None,
           remove_connections: Optional[List[str]] = None) -> bool:
    """
    Update an existing node in memory.
    
    Args:
        node_id: The ID of the node to update
        metadata: New metadata (if string, treated as content)
        add_connections: New connections to add (list of node IDs)
        remove_connections: Connection IDs to remove
    
    Returns:
        True if the update was successful, False otherwise
    """
    node = _memory_graph.get_node(node_id)
    if not node:
        return False
    
    # Update metadata if provided
    if metadata is not None:
        old_hash = _memory_graph._get_hash(node.metadata)
        
        # Convert string metadata to dict with 'content' key
        if isinstance(metadata, str):
            node.metadata = {'content': metadata}
        elif isinstance(metadata, dict):
            node.metadata = metadata
        
        # Update the content_hash_to_id map
        if old_hash in _memory_graph.content_hash_to_id_map:
            del _memory_graph.content_hash_to_id_map[old_hash]
        new_hash = _memory_graph._get_hash(node.metadata)
        _memory_graph.content_hash_to_id_map[new_hash] = node_id
    
    # Add new connections
    if add_connections:
        for target_id in add_connections:
            if target_id and target_id in _memory_graph.nodes:
                node.add_edge(target_id)
    
    # Remove connections
    if remove_connections:
        for target_id in remove_connections:
            node.remove_edge(target_id)
    
    return True
