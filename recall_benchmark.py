# recall benchmark: how often does HNSW find the TRUE top 5 (brute force) and how fast?
# run with:  python recall_benchmark.py
# no embedding model needed, random vectors behave the same way for this test

import random
import time

from search import search
from hnsw import insert, layered_search

DIMS = 32
QUERIES = 50


def make_vectors(n, rng):
    # centered on zero so cosine similarity spreads out between -1 and 1
    return [(f"item{i}", [rng.random() - 0.5 for _ in range(DIMS)]) for i in range(n)]


def run(n):
    rng = random.Random(7)
    catalog = make_vectors(n, rng)

    graph, layers = {}, {}
    start = time.time()
    for name, vector in catalog:
        insert(graph, layers, name, vector)
    build_time = time.time() - start
    entry = max(layers, key=layers.get)  # search starts from the node on the highest floor

    print(f"\n{n} vectors, {DIMS} dims (index built in {build_time:.1f}s)")
    print(f"{'ef':>5} {'recall@5':>10} {'HNSW ms':>9} {'brute ms':>9}")
    for ef in (10, 25, 50, 100):
        query_rng = random.Random(99)  # the same queries for every ef
        hits = hnsw_time = brute_time = 0
        for _ in range(QUERIES):
            query = [query_rng.random() - 0.5 for _ in range(DIMS)]
            t = time.perf_counter()
            truth = {name for name, _ in search(query, catalog, k=5)}
            brute_time += time.perf_counter() - t
            t = time.perf_counter()
            found = {name for name, _ in layered_search(graph, query, layers, start=entry, k=5, ef=ef)}
            hnsw_time += time.perf_counter() - t
            hits += len(truth & found)
        print(f"{ef:>5} {hits / (5 * QUERIES):>10.3f} {1000 * hnsw_time / QUERIES:>9.2f} {1000 * brute_time / QUERIES:>9.2f}")


if __name__ == "__main__":
    for n in (1000, 5000):
        run(n)
