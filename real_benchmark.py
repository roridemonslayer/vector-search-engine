from sentence_transformers import SentenceTransformer
from search import search
from hnsw import insert, layered_search

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "chunky knit sweater for cold days",
    "floral summer dress",
    "black leather combat boots",
    "soft pastel hoodie",
    "waterproof hiking jacket",
    "denim button-up shirt",
    "silk slip dress",
    "wool peacoat",
    "graphic band tee",
    "linen summer shorts",
]

#building a real catalog with name and real emebeding vector 

catalog = []
for t in texts: 
    vector = model.encode(t)
    catalog.append((t,vector))

g = {}
l = {}
for name, vector in catalog: 
    insert(g, l, name, vector)

# ---- try a real search phrase ----
query_text = "something warm and cozy for winter"
query_vector = model.encode(query_text)

print("Query:", query_text)
print()
print("Brute-force top 5:")
for name, score in search(query_vector, catalog, k=5):
    print(f"  {name}  ({score:.3f})")

print()
print("HNSW top 5:")
for name, score in layered_search(g, query_vector, l, start=texts[0], k=5):
    print(f"  {name}  ({score:.3f})")

