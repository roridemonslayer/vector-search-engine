from sentence_transformers import SentenceTransformer #important liberty 


model = SentenceTransformer("all-MiniLM-L6-v2") # We're grabbing the specfic model 

vec = model.encode("black graphic tee") # this is saying give me the vector of this phrase

print(len(vec))
print(vec[:5])

