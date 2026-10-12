from search import cosine
from search import search
import heapq
import pickle
import random

graph = {
    "1": {"vector": [1, 8], "neighbors": ["2", "4"]},
    "2": {"vector": [3, 7], "neighbors": ["1", "6", "3"]},
    "3": {"vector": [4, 8], "neighbors": ["2"]},
    "4": {"vector": [0, 3], "neighbors": ["1"]},
    "5": {"vector": [7, 1], "neighbors": ["6"]},
    "6": {"vector": [8, 2], "neighbors": ["2", "5"]},
}

def assign_layer():
    layer  = 0 #starts at layer 0 

    while random.random() < 0.5: #50% chance to keep climbing 
        layer+=1 #go up one more interval
    return layer


layers = {}
for node in graph:
    layers[node] = assign_layer() 

def neighbors_at_layer(node, layer_num, layers, graph ):
    all_neighbors = graph[node]["neighbors"]
    valid = []
    for n in all_neighbors:
        if layers[n]>= layer_num:
            valid.append(n)
    return valid



def greedy_search(graph, query, start = "1"): #the start of our function 
    current = start #where we're starting out rn 
    while True: #keep hopping with till we have a return statement
        #Step 1: how good is OUR current spot?
        # We look up our own vector and score it against the query.
        # Higher cosine score = closer in meaning = better.
        current_vector = graph[current]["vector"] #current node point 
        current_score = cosine(query,current_vector) #how close we are to the query
        # Step 2: set up a "best so far" tracker.
        # We start by assuming nobody beats us the best_score is just our own score.
        # best_neighbor stays None unless someone actually wins.

        best_neighbor = None
        best_score = current_score


        # Step 3: check every node we're directly connected to (our "friends").
        # This is the whole point of HNSW , we only check OUR neighbors,
        # never the entire graph.
        for neighbors in graph[current]["neighbors"]:
            neighbor_vector = graph[neighbors]["vector"]
            neighbor_score = cosine(query, neighbor_vector)

            # If this neighbor is closer to the query than our current best,
            # they become the new leader.
            if neighbor_score > best_score:
                best_score = neighbor_score
                best_neighbor = neighbors
            #If NO neighbor beat our current spot, we're at a "local best."
        # Nothing nearby can improve on us, so we stop and report where we are.
        if best_neighbor is None:
            return current
        #Otherwise, someone nearby WAS closer, so we move there and
        # the whole process repeats from the new position.

        current = best_neighbor

def search_layer(graph, query, entry_points, ef, layer_num, layers):
    # this is the "beam" version of greedy_search, the way the original HNSW paper does it.
    # greedy_search follows ONE path and stops at the first dead end.
    # here we keep the ef best nodes we've seen so far and keep exploring from all of them,
    # so one bad turn early on doesn't decide the whole answer.

    visited = set(entry_points)
    # candidates = nodes we still need to explore from (a heap, best first)
    # found = the ef best nodes so far (a heap, worst first, so the worst is easy to kick out)
    candidates = []
    found = []
    for node in entry_points:
        score = cosine(query, graph[node]["vector"])
        heapq.heappush(candidates, (-score, node))  # minus because heapq pops the smallest first
        heapq.heappush(found, (score, node))

    while candidates:
        neg_score, node = heapq.heappop(candidates)
        worst_found = found[0][0]
        # if the best thing left to explore is already worse than everything we kept, we're done
        if -neg_score < worst_found and len(found) >= ef:
            break
        for neighbor in neighbors_at_layer(node, layer_num, layers, graph):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            score = cosine(query, graph[neighbor]["vector"])
            if len(found) < ef or score > found[0][0]:
                heapq.heappush(candidates, (-score, neighbor))
                heapq.heappush(found, (score, neighbor))
                if len(found) > ef:
                    heapq.heappop(found)  # drop the worst so we only ever keep ef

    return sorted(((node, score) for score, node in found), key=lambda pair: pair[1], reverse=True)


def layered_search(graph, query, layers, start = "1", k = 5, ef = 50):
    top_layer = max(layers.values()) # find the highest floor that exists in our building
    current = start #where we're standing at right no w
    current_layer = top_layer #the room we;re searching for

    # upper floors: same greedy hopping as before, just to land somewhere close to the query fast
    while current_layer > 0:
        while True: #this inner loiop is the actual hop, 
            current_score = cosine(query, graph[current]["vector"])# how close we are the query righ tnow 
            best_neighbor = None #the floor nobody has beatedn us yet on 
            best_score = current_score# score to bea t= our own score
            for neighbor in neighbors_at_layer(current, current_layer, layers, graph): #only looks at the neigh who are allowed on thsi spseicf c floor
                neighbor_score = cosine(query, graph[neighbor]["vector"]) #how close they are to the query
                if neighbor_score > best_score: #did this neighbor beat oru curren t best one 
                    best_score = neighbor_score #update the score
                    best_neighbor = neighbor #remeber who won
            if best_neighbor is None: #if no one on the floor was close than us
                break #stop hooping on the floor and go check downstairs
            current = best_neighbor #someone was closer keep hoping htere
        current_layer -= 1# doen with htusi floor go doen oner lleve

    # ground floor: the beam search. this is where the actual top k comes from.
    # before this, the answer was only the handful of nodes on one greedy path,
    # which is why recall was so low.
    return search_layer(graph, query, [current], max(ef, k), 0, layers)[:k]


def insert(graph, layers, name, vector, k = 8, ef_construction = 40, max_neighbors = 16):
    # k = how many neighbors a new node links to (the paper calls this M)
    # ef_construction = how wide the beam is while looking for those neighbors
    # max_neighbors = the most links any node keeps. without a cap, popular nodes
    # end up with hundreds of links and every search slows down.

    #this gives the new node a random layer
    node_layer = assign_layer()

    #this is a special case . if thi sis the first node, then theres nothign to search 
    if not graph:
        layers[name] = node_layer
        graph[name] = {"vector": vector, "neighbors": []}
        return

    # small graphs: just check everyone (same as the old version, using our own search())
    if len(graph) <= ef_construction:
        catalog = [(existing, graph[existing]["vector"]) for existing in graph]
        closest = search(vector, catalog, k = k)
    else:
        # bigger graphs: find the neighbors with the graph's OWN search instead of checking
        # every node. this is what makes building the index fast at scale.
        entry = max(layers, key = layers.get)  # start from the node on the highest floor
        closest = layered_search(graph, vector, layers, start = entry, k = k, ef = ef_construction)

    layers[name] = node_layer
    graph[name] = {"vector": vector, "neighbors": []}

    #connect both directions  for each lose match 
    for match_name, score in closest:
        graph[name]["neighbors"].append(match_name)
        graph[match_name]["neighbors"].append(name)
        # if that neighbor now has too many links, keep only its closest ones
        if len(graph[match_name]["neighbors"]) > max_neighbors:
            prune(graph, match_name, max_neighbors)


def prune(graph, node, max_neighbors):
    # keep the max_neighbors links that point to the most similar nodes
    vector = graph[node]["vector"]
    ranked = sorted(graph[node]["neighbors"], key = lambda other: cosine(vector, graph[other]["vector"]), reverse = True)
    graph[node]["neighbors"] = ranked[:max_neighbors]


def save_index(graph, layers,filepath):
    data = {"graph":graph, "layers" :layers}
    with open(filepath, "wb") as f:
        pickle.dump(data, f) 

def load_index(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["graph"], data["layers"]

# the demo only runs when this file is run directly (python hnsw.py).
# before this guard, just importing hnsw printed everything and rewrote test_index.pkl
if __name__ == "__main__":
    query = [7,1]      # sits right on top of node 3's vector
    print(greedy_search(graph, query, start="1"))
    for i in range(20):
        print(assign_layer())


    print(layers)

    query = [4, 8]
    print(layered_search(graph, query, layers, start="1"))

    new_graph = {}
    new_layers = {}

    insert(new_graph, new_layers, "a", [1, 8])
    insert(new_graph, new_layers, "b", [3, 7])
    insert(new_graph, new_layers, "c", [4, 8])

    print(new_graph)



    save_index(new_graph, new_layers, "test_index.pkl")

    loaded_graph, loaded_layers = load_index("test_index.pkl")
    print(loaded_graph == new_graph)      # should print True
    print(loaded_layers == new_layers)    # should print True
