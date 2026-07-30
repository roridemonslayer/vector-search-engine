from hnsw import load_index

graph, layers = load_index("test_index.pkl")
print(graph)
print(layers)