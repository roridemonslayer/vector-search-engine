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

- **Graph construction** — nodes are inserted one at a time via `insert()`. Each new node finds its 8 closest neighbors (brute force while the graph is small, then the graph's own beam search), connects to them in both directions, and `prune()` caps every node at its 16 closest links.
- **Layers** — each node is randomly assigned a "height" via a weighted coin-flip (`assign_layer()`), the same mechanism the original HNSW paper uses. Most nodes stay at the ground floor; a rare few climb higher and become long-range shortcuts.
- **Search** — starts at the top layer, greedily hops toward the query, drops a floor when stuck, and repeats; on the ground floor it switches to a beam search that keeps the `ef` best candidates, which is where the top k comes from.
- **Persistence** — the graph and layer assignments can be saved to disk (`save_index`) and reloaded (`load_index`) via `pickle`, so the index doesn't need to be rebuilt from scratch on every run.

## Benchmarks

How often HNSW finds the **true** top 5 (from brute force), and how fast, on random 32-dimensional vectors. `ef` is the width of the ground-floor beam: bigger means more accurate but slower.

```
1000 vectors, 32 dims (index built in 1.7s)
   ef   recall@5   HNSW ms  brute ms
   10      0.648      1.07      5.56
   25      0.864      1.48      4.70
   50      0.976      2.40      4.86
  100      0.996      3.95      5.27

5000 vectors, 32 dims (index built in 18.8s)
   ef   recall@5   HNSW ms  brute ms
   10      0.420      1.31     25.86
   25      0.736      2.60     30.60
   50      0.848      4.33     32.11
  100      0.952      6.90     30.50
```

- At 5,000 vectors, HNSW is **4–20× faster** than brute force depending on `ef`, and the gap keeps growing with more data, because brute force checks every vector while HNSW only checks a small neighborhood.
- `ef=50` hits **97.6% recall** at 1,000 vectors. At 5,000, raising `ef` to 100 gets back to 95%.

**What changed:** the first version scored **3.2% recall@5** on the same 1,000 vectors. Two fixes from the original HNSW paper closed the gap:
1. **Beam search on the ground floor.** Search used to follow one greedy path and return only the nodes it passed. Now it keeps the `ef` best candidates and explores from all of them (`search_layer`).
2. **Better construction.** New nodes find neighbors with the graph's own search instead of brute force, link to 8 neighbors instead of 2, and every node's links are pruned to its 16 closest (`prune`).

Full benchmark script: `recall_benchmark.py`

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
├── hnsw.py              # HNSW graph: assign_layer, insert + prune, greedy_search, search_layer (beam), layered_search, save/load
├── real_benchmark.py     # end-to-end demo on real text embeddings
├── recall_benchmark.py   # HNSW recall@5 and speed vs brute force, across ef values
├── mock_benchmark.py     # first speed comparison on fake vectors
├── test_engine.py        # unit tests (python -m unittest -v)
├── embed_test.py          # first exploration of the embedding model
├── real_search.py          # early real-embedding search test
└── README.md
```

## What I learned building this

- Why raw distance metrics (L1/Euclidean) can fail on real embeddings, and why cosine similarity is the industry standard
- How HNSW's layered graph structure trades a small amount of accuracy for a large speed gain — and why that tradeoff is called "approximate nearest neighbor" search
- That an unreachable node in a graph is invisible to greedy search, no matter how good the search logic is — connectivity determines what's findable
- How to build, benchmark, and honestly evaluate an algorithm's real limitations rather than just claiming it works
- That following a single greedy path throws away almost everything the graph knows; a beam of candidates is what turns HNSW from a toy into a real index

## Future work

- ~~Multi-candidate neighbor selection during insertion~~ done: recall@5 went from 3.2% to 97.6%
- Benchmark at larger scale (50k–100k+ vectors); 5k already shows the speed gap growing
- Per-layer neighbor lists (the paper keeps a separate list per floor; this version shares one)
- Make the embedding model swappable as a parameter
- Batch insertion for faster index construction

## Stack

Python, [`sentence-transformers`](https://www.sbert.net/) (`all-MiniLM-L6-v2`), `pickle` for persistence. No FAISS, no Chroma, no vector database dependencies — every distance calculation, graph structure, and search algorithm is implemented from scratch.