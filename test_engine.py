# tests for the engine — run with:  python -m unittest -v
# every expected answer here is small enough to check by hand

import os
import random
import tempfile
import unittest

from search import cosine, dot, gap_sum, magnitude, search
from hnsw import assign_layer, insert, layered_search, load_index, save_index


class VectorMathTests(unittest.TestCase):
    def test_dot(self):
        # (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
        self.assertEqual(dot([1, 2, 3], [4, 5, 6]), 32)

    def test_magnitude(self):
        # a 3-4-5 triangle: sqrt(3*3 + 4*4) = sqrt(25) = 5
        self.assertEqual(magnitude([3, 4]), 5)

    def test_gap_sum(self):
        # |1-4| + |2-0| = 3 + 2 = 5
        self.assertEqual(gap_sum([1, 2], [4, 0]), 5)

    def test_cosine_same_direction_is_one(self):
        # [2, 4] is just [1, 2] stretched, same direction -> cosine 1, even though the lengths differ
        self.assertAlmostEqual(cosine([1, 2], [2, 4]), 1.0)

    def test_cosine_perpendicular_is_zero(self):
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)

    def test_cosine_opposite_is_minus_one(self):
        self.assertAlmostEqual(cosine([1, 1], [-1, -1]), -1.0)


class BruteForceSearchTests(unittest.TestCase):
    catalog = [
        ("tee A", [0.9, 0.8, 0.1]),
        ("tee b", [0.8, 0.9, 0.1]),
        ("gown", [0.1, 0.8, 0.9]),
    ]

    def test_closest_item_comes_first(self):
        top = search([0.85, 0.9, 0.05], self.catalog, k=1)
        self.assertEqual(top[0][0], "tee b")

    def test_returns_k_results_best_first(self):
        results = search([0.85, 0.9, 0.05], self.catalog, k=3)
        self.assertEqual(len(results), 3)
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(results[-1][0], "gown")


class HNSWTests(unittest.TestCase):
    def test_assign_layer_is_usually_zero(self):
        # each climb is a 50% coin flip, so about half of all nodes stay on layer 0
        random.seed(0)
        layers = [assign_layer() for _ in range(2000)]
        share_on_ground = layers.count(0) / len(layers)
        self.assertTrue(0.45 < share_on_ground < 0.55)

    def test_insert_connects_both_directions(self):
        graph, layers = {}, {}
        insert(graph, layers, "a", [1, 8])
        insert(graph, layers, "b", [3, 7])
        self.assertIn("b", graph["a"]["neighbors"])
        self.assertIn("a", graph["b"]["neighbors"])

    def test_search_finds_an_exact_match(self):
        random.seed(1)
        graph, layers = {}, {}
        for name, vector in [("a", [1, 8]), ("b", [3, 7]), ("c", [4, 8]), ("d", [8, 2]), ("e", [7, 1])]:
            insert(graph, layers, name, vector)
        top = layered_search(graph, [7, 1], layers, start="a", k=1)
        self.assertEqual(top[0][0], "e")

    def test_save_and_load_round_trip(self):
        graph, layers = {}, {}
        insert(graph, layers, "a", [1, 8])
        insert(graph, layers, "b", [3, 7])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "index.pkl")
            save_index(graph, layers, path)
            loaded_graph, loaded_layers = load_index(path)
        self.assertEqual(loaded_graph, graph)
        self.assertEqual(loaded_layers, layers)


if __name__ == "__main__":
    unittest.main()
