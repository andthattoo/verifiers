from typing import Dict, List, Optional, Tuple, Set, Any, Union
import uuid
import numpy as np
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class MemoryNode:
    def __init__(self, metadata: Union[str, Dict[str, Any]]):
        self.id = str(uuid.uuid4())

        # Convert string metadata to a dict with 'content' key
        if isinstance(metadata, str):
            self.metadata = {"content": metadata}
        else:
            self.metadata = metadata or {}

        self.edges = set()  # Set of connected node IDs

    def add_edge(self, target_node_id: str):
        self.edges.add(target_node_id)

    def remove_edge(self, target_node_id: str):
        if target_node_id in self.edges:
            self.edges.remove(target_node_id)

    def get_text_content(self) -> str:
        """Get text representation of node for search purposes"""
        if "content" in self.metadata:
            return self.metadata["content"]

        # If no content field, concatenate all string values
        return " ".join(str(v) for v in self.metadata.values()
                        if isinstance(v, (str, int, float)))


class MemoryGraph:
    def __init__(self):
        self.nodes = {}  # Map of id -> node
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.content_hash_to_id_map = {}  # For deduplication
        self.tokenize_pattern = re.compile(r'\w+')

    def _get_hash(self, metadata: Union[str, Dict[str, Any]]) -> str:
        """Generate a simple hash for deduplication"""
        if isinstance(metadata, str):
            return metadata
        if isinstance(metadata, dict) and "content" in metadata:
            return str(metadata["content"])
        return str(metadata)

    def add_node(self, metadata: Union[str, Dict[str, Any]]) -> str:
        content_hash = self._get_hash(metadata)

        # Check for duplicates
        if content_hash in self.content_hash_to_id_map:
            return self.content_hash_to_id_map[content_hash]

        node = MemoryNode(metadata)
        self.nodes[node.id] = node
        self.content_hash_to_id_map[content_hash] = node.id
        return node.id

    def add_edge(self, source_id: str, target_id: str) -> bool:
        if source_id not in self.nodes or target_id not in self.nodes:
            return False

        self.nodes[source_id].add_edge(target_id)
        return True

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        return self.nodes.get(node_id)

    def get_all_text_contents(self) -> List[str]:
        return [node.get_text_content() for node in self.nodes.values()]

    def bm25_setup(self):
        """Set up BM25 with current nodes"""
        self.corpus = [self.tokenize_pattern.findall(node.get_text_content().lower())
                       for node in self.nodes.values()]
        self.id_list = list(self.nodes.keys())
        self.bm25 = BM25Okapi(self.corpus)

    def bm25_search(self, query: str, top_k: int = 5) -> List[str]:
        """Search nodes using BM25"""
        if not hasattr(self, 'bm25') or len(self.corpus) != len(self.nodes):
            self.bm25_setup()

        tokenized_query = self.tokenize_pattern.findall(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:]
        return [self.id_list[i] for i in top_indices]

    def semantic_search(self, query: str, candidate_ids: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Perform semantic search on candidate nodes"""
        if not candidate_ids:
            return []

        query_embedding = self.model.encode(query)
        candidates = [(node_id, self.nodes[node_id].get_text_content()) for node_id in candidate_ids]
        candidate_embeddings = self.model.encode([c[1] for c in candidates])

        # Calculate cosine similarity
        similarities = np.dot(candidate_embeddings, query_embedding) / (
                np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Create (node_id, similarity) tuples
        results = [(candidates[i][0], float(similarities[i])) for i in range(len(candidates))]

        # Sort by similarity score and take top_k
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def bfs_exploration(self, start_nodes: List[str], max_depth: int = 2) -> Set[str]:
        """
        Perform BFS from start nodes up to max_depth.
        Returns a set of node IDs.
        """
        discovered = set(start_nodes)
        queue = [(node_id, 0) for node_id in start_nodes]  # (node_id, depth)

        while queue:
            current_id, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            current_node = self.nodes.get(current_id)
            if not current_node:
                continue

            for neighbor_id in current_node.edges:
                if neighbor_id not in discovered:
                    discovered.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

        return discovered