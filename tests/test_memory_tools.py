"""
Tests for memory tools implementation.
"""
import unittest
from verifiers.tools.memory_tools import write, search, update, _memory_graph


class TestMemoryTools(unittest.TestCase):
    def setUp(self):
        # Clear memory graph before each test
        _memory_graph.nodes = {}
        _memory_graph.content_hash_to_id_map = {}
        
    def test_write_and_search_with_string(self):
        # Write some nodes using string metadata
        node1_id = write("This is some information about machine learning.")
        node2_id = write("Python is a programming language widely used in AI.")
        node3_id = write("Deep learning is a subset of machine learning.")
        
        # Test basic search
        results = search("machine learning")
        self.assertGreaterEqual(len(results), 2)
        
        # Check that node1 and node3 are in results (they mention machine learning)
        result_ids = [r["id"] for r in results]
        self.assertIn(node1_id, result_ids)
        self.assertIn(node3_id, result_ids)
        
    def test_write_and_search_with_dict(self):
        # Write some nodes using dictionary metadata
        node1_id = write({"content": "Information about machine learning", "category": "ML"})
        node2_id = write({"content": "Python programming language", "category": "Programming"})
        
        # Test search
        results = search("machine learning")
        self.assertGreaterEqual(len(results), 1)
        
        # Check node1 is in results
        result_ids = [r["id"] for r in results]
        self.assertIn(node1_id, result_ids)
        
        # Verify metadata is returned correctly
        for result in results:
            if result["id"] == node1_id:
                self.assertEqual(result["metadata"]["category"], "ML")
            elif result["id"] == node2_id:
                self.assertEqual(result["metadata"]["category"], "Programming")
        
    def test_connections(self):
        # Write some nodes with connections
        node1_id = write("Neural networks are used in deep learning.")
        node2_id = write("Deep learning is a subset of machine learning.")
        
        # Connect node1 to node2
        write("Connecting information", connections=[node1_id, node2_id])
        
        # Search should now find related information through connections
        results = search("neural networks machine learning", bfs_depth=2)
        self.assertGreaterEqual(len(results), 2)
        
    def test_update_with_string(self):
        # Write a node
        node_id = write({"content": "Initial content", "key": "value"})
        
        # Update the node with a string
        success = update(node_id, "Updated content")
        self.assertTrue(success)
        
        # Check that the update worked
        results = search("Updated content")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], node_id)
        self.assertEqual(results[0]["metadata"]["content"], "Updated content")
        
    def test_update_with_dict(self):
        # Write a node
        node_id = write("Initial content")
        
        # Update the node with a dict
        success = update(node_id, {"content": "Updated content", "new_key": "new_value"})
        self.assertTrue(success)
        
        # Check that the update worked
        results = search("Updated content")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], node_id)
        self.assertEqual(results[0]["metadata"]["content"], "Updated content")
        self.assertEqual(results[0]["metadata"]["new_key"], "new_value")
        
    def test_add_and_remove_connections(self):
        # Create nodes
        node1_id = write("First node")
        node2_id = write("Second node")
        node3_id = write("Third node")
        
        # Test adding connections
        update(node1_id, add_connections=[node2_id, node3_id])
        node = _memory_graph.get_node(node1_id)
        self.assertEqual(len(node.edges), 2)
        self.assertIn(node2_id, node.edges)
        self.assertIn(node3_id, node.edges)
        
        # Test removing connections
        update(node1_id, remove_connections=[node2_id])
        self.assertEqual(len(node.edges), 1)
        self.assertNotIn(node2_id, node.edges)
        self.assertIn(node3_id, node.edges)
        
    def test_bm25_and_semantic_search(self):
        # Write some nodes with varying content
        write("Machine learning is a field of study.")
        write("Python programming language is used for AI.")
        write("TensorFlow and PyTorch are ML frameworks.")
        write("Deep learning involves neural networks.")
        write("Computer vision is a machine learning application.")
        
        # Test that search combines BM25 and semantic results
        results = search("neural network frameworks for deep learning")
        self.assertGreaterEqual(len(results), 2)
        
        # The top result should have "neural networks" or "frameworks" mentioned
        top_result_text = results[0]["metadata"]["content"].lower()
        self.assertTrue("neural networks" in top_result_text or "frameworks" in top_result_text)
        
    def test_deduplication(self):
        # Write the same content twice
        id1 = write("This is duplicate content")
        id2 = write("This is duplicate content")
        
        # They should have the same ID (deduplication)
        self.assertEqual(id1, id2)
        
        # Search should only return one result
        results = search("duplicate content")
        self.assertEqual(len(results), 1)
        
    def test_search_parameters(self):
        # Write several nodes
        for i in range(10):
            write(f"Node {i} content with varying information")
            
        # Test different search parameters
        # Basic search
        results1 = search("content", top_k=3)
        self.assertLessEqual(len(results1), 3)
        
        # Wider BM25 search
        results2 = search("content", bm25_candidates=8, top_k=5)
        
        # Deeper BFS
        results3 = search("content", bfs_depth=2, top_k=5)
        
        # All parameters should work without errors
        self.assertTrue(isinstance(results1, list))
        self.assertTrue(isinstance(results2, list))
        self.assertTrue(isinstance(results3, list))


if __name__ == "__main__":
    unittest.main()