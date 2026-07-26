from sentence_transformers import SentenceTransformer #pulling model in here 
# basically importanting the engine
from search import gap_sum, search

#loading model 

model = SentenceTransformer("all-MiniLM-L6-v2")

#actual real time descriptions
texts = [
    "chunky knit sweater for cold days",
    "floral summer dress",
    "black leather combat boots",
    "soft pastel hoodie",
    "waterproof hiking jacket",
]

catalog = []
for t in texts:
    vector = model.encode(t) # getting the vector decimeals from the t 
    catalog.append((t, vector))

queries = [
    "warm knitwear for cold weather",
    "shoes for rough trails",
    "something for a rainy hike",
]

for q in queries:
    query = model.encode(q)
    result = search(query, catalog, k=2)
    print(q, "→", result)