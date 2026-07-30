#this is the benchmark script
#we;re buildng this part to prove that Prove, with real numbers, that your HNSW is actually faster than brute-force search

import time #allows us to measure time 
import random #generates fake. test_vectors
from search import search  #The bruter force engine
from hnsw import graph, layers, insert, layered_search # the HNSW pieces 


# STEP 1: generate a batch of fake vectors to test with 
# We don't need real embeddings for a speed test, we just need a random numbers behave
# the same way, timing-wise.
# tests like recall, less for raw speed tests.

def make_fake_vectors(n, dims = 8):
    #n = how many fake vectors to make 
    #dim = how many nm are in each vector
    vectors = []
    for i in range(n):
        v = [random.random() for _ in range(dims)] 
        #this line creates one fake vector: and a list of random dims 
        #random,random() gives a random number between 0.0 - 1.0

        vectors.append((f"item{i}", v))
    return vectors

#this will time the brute force search 
def time_brute_force(query, catalog , k = 5):
    start = time.time()# this snapsshpots the clock befroe it starts 
    result= search(query, catalog, k = k) #this runs the brute force engine
    end = time.time() #snapsshots the clock after
    elapsed = end - start #how long the search took
    return result, elapsed #shows both the answer and how long it took 


#building the HNSW graph from the same catalog

def build_hnsw(catalog):
    g = {} #empty graph no nodes 
    l = {} #empty layer dict to match 
    for name , vector in catalog: 
        insert(g, l, name, vector) #resuses the insert function 
    return g, l

#timing the hnsw graph 

def time_hnsw(query,g, l, start_node):
    start  = time.time()
    result_node = layered_search(g, query, l, start = start_node) #running the layered search 
    end = time.time()
    elapsed = end - start #how long the search took
    return result_node, elapsed


#running the experiemene t
catalog = make_fake_vectors(200)              # start small — 200 fake items, 8 dims each
query = [random.random() for _ in range(8)]    # one random query vector, same shape as the catalog

# time brute-force first — this is our "ground truth" baseline
brute_result, brute_time = time_brute_force(query, catalog, k=5)
print("Brute-force top 5:", brute_result)
print("Brute-force time:", brute_time)

# now build the HNSW graph (this itself takes time — inserting 200 nodes one by one)
g, l = build_hnsw(catalog)

# time HNSW's search on that same query
hnsw_result, hnsw_time = time_hnsw(query, g, l, start_node=catalog[0][0])
print("HNSW result:", hnsw_result)
print("HNSW time:", hnsw_time)


