# vector-search-engine

A vector search engine built completely from scratch in Python — semantic search by meaning, not keywords. No FAISS, no Chroma, no shortcuts.

Search for `"something warm and cozy for winter"` and it finds `"chunky knit sweater for cold days"` — zero shared words. That's the whole point: this engine matches on *meaning*, not exact text.

This project was built to understand what vector databases actually do under the hood, one piece at a time: distance metrics, cosine similarity, HNSW graph indexing, real AI embeddings, persistence, and benchmarking.

## What's inside

The engine has two search implementations, built in this order:

**1. Brute-force search** (`search.py`)
Compares a query against every stored vector, one at a time, using cosine similarity. Always exact — guaranteed to find the true closest matches. Used as the "ground truth" baseline for everything else.

**2. HNSW (Hierarchical Navigable Small World) search** (`hnsw.py`)
An approximate nearest-neighbor index inspired by the algorithm used in FAISS, Pinecone, and other production vector databases. Instead of checking every vector, it builds a graph where each vector connects to a handful of nearby "neighbors," organized into layers — a sparse top layer for long-range jumps, denser lower layers for precision. Search hops toward the query instead of scanning everything.

## How it works

```
text  →  embedding model (sentence-transformers)  →  384-dim vector
                                                            |
                                    query vector  ──────────┤
                                                            |
                              brute-force search    OR    HNSW search
                                     |                        |
                              exact top-k              approximate top-k
```

### The math

- **Cosine similarity** — measures the *angle* between two vectors, ignoring their size. This was a deliberate choice after an earlier version (using raw L1/"gap-sum" distance) ranked results incorrectly on real embeddings — it was punishing vectors for being different *lengths*, not different *meanings*. Cosine fixed it by comparing direction only.
- **Dot product & magnitude** — the two building blocks cosine similarity is made of, both implemented from scratch.

### The HNSW index

- **Graph construction** — nodes are inserted one at a time via `insert()`. Each new node searches the existing graph (using the engine's own brute-force `search()`) to find its closest neighbors, then connects to them in both directions.
- **Layers** — each node is randomly assigned a "height" via a weighted coin-flip (`assign_layer()`), the same mechanism the original HNSW paper uses. Most nodes stay at the ground floor; a rare few climb higher and become long-range shortcuts.
- **Search** — starts at the top layer, greedily hops toward the query, drops a floor when stuck, and repeats until it bottoms out at the full graph.
- **Persistence** — the graph and layer assignments can be saved to disk (`save_index`) and reloaded (`load_index`) via `pickle`, so the index doesn't need to be rebuilt from scratch on every run.

## Benchmarks

Brute-force vs. HNSW, tested on 200 randomly generated 8-dimensional vectors:

| Metric | Brute-force | HNSW |
|---|---|---|
| Search time | ~0.0004s | ~0.0001s |
| Speedup | — | **~3–4x faster** |
| Recall@5 (vs. brute-force ground truth) | 100% (exact) | ~25% |

**Honest note on recall:** this HNSW implementation trades a meaningful amount of accuracy for its speed gain at small scale. The likely cause is a simplified single-pass neighbor selection during insertion — production HNSW implementations connect new nodes to *multiple* candidate neighbors and prune weaker edges over time, which this from-scratch version doesn't yet do. That refinement is a natural next step (see below).

Full benchmark script: `benchmark.py`

## Real-text demo

Using `sentence-transformers` (`all-MiniLM-L6-v2`) to embed real product descriptions:

```
Query: "something warm and cozy for winter"

Brute-force top match: "chunky knit sweater for cold days"  (0.545)
```

No word overlap between the query and the result — the match is purely semantic. Script: `real_benchmark.py`

## Project structure

```
vector-search-engine/
├── search.py           # brute-force engine: gap_sum (L1), dot, magnitude, cosine, search()
├── hnsw.py              # HNSW graph: assign_layer, insert, greedy_search, layered_search, save/load
├── real_benchmark.py     # end-to-end demo on real text embeddings
├── benchmark.py          # brute-force vs HNSW speed + recall comparison
├── embed_test.py          # first exploration of the embedding model
├── real_search.py          # early real-embedding search test
└── README.md
```

## What I learned building this

- Why raw distance metrics (L1/Euclidean) can fail on real embeddings, and why cosine similarity is the industry standard
- How HNSW's layered graph structure trades a small amount of accuracy for a large speed gain — and why that tradeoff is called "approximate nearest neighbor" search
- That an unreachable node in a graph is invisible to greedy search, no matter how good the search logic is — connectivity determines what's findable
- How to build, benchmark, and honestly evaluate an algorithm's real limitations rather than just claiming it works

## Future work

- Multi-candidate neighbor selection during insertion (closer to the original HNSW paper's construction algorithm), to improve recall
- Benchmark at larger scale (10k–100k+ vectors) to show HNSW's speed advantage grow with dataset size
- Make the embedding model swappable as a parameter
- Batch insertion for faster index construction

## Stack

Python, [`sentence-transformers`](https://www.sbert.net/) (`all-MiniLM-L6-v2`), `pickle` for persistence. No FAISS, no Chroma, no vector database dependencies — every distance calculation, graph structure, and search algorithm is implemented from scratch.