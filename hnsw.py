from search import cosine
from search import search
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

def neighbors_at_layer(node, layer_num, layers):
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

def layered_search(graph, query, layers, start = "1"):
    top_layer = max(layers.values()) # find the highest floor that exists in our building
    current = start #where we're standing at right no w
    current_layer = top_layer #the room we;re searching for
    while current_layer >= 0: #this keeps the search going as long as theres a floor left to check 
        while True: #this inner loiop is the actual hop, 
            current_vector = graph[current]["vector"] #the current spots location
            current_score = cosine(query,current_vector)# how close we are the query righ tnow 

            best_neighbor = None #the floor nobody has beatedn us yet on 
            best_score = current_score# score to bea t= our own score

            for neighbor in neighbors_at_layer(current, current_layer, layers): #only looks at the neigh who are allowed on thsi spseicf c floor
                neighbor_vector = graph[neighbor]["vector"] #the neighbors location 
                neighbor_score = cosine(query, neighbor_vector) #how close they are to the query

                if neighbor_score > best_score: #did this neighbor beat oru curren t best one 
                    best_score = neighbor_score #update the score
                    best_neighbor = neighbor #remeber who won
            if best_neighbor is None: #if no one on the floor was close than us
                break #stop hooping on the floor and go check downstairs
            current = best_neighbor #someone was closer keep hoping htere
        current_layer -= 1# doen with htusi floor go doen oner lleve
    return current


def insert(graph, layers, name, vector, k = 2): 
    #this gives the new node a random layer
    node_layer = assign_layer()
    layers[name] = node_layer

    #add hte node to the graph with an empty neighbor list 
    graph[name] =  {"vector": vector, "neighbors":[]}

    #this is a special case . if thi sis the first node, then theres nothign to search 
    if len(graph) == 1:
        return

    #search() expects a list of (name, vector) pairs — same shape as unit 4's catalog
    catalog = []
    for existing_name in graph:
        if existing_name != name:
            existing_vector = graph[existing_name]["vector"]
            catalog.append((existing_name, existing_vector))

    #finds the k closet  existing nodes using the  search()
    closest = search(vector, catalog, k = k)
    # closest looks like: [("3", 0.91), ("6", 0.75)], (name, score) pairs

    #connect both directions  for each lose match 
    for match_name , score in closest:
        graph[name]["neighbors"].append(match_name) 
        graph[match_name]["neighbors"].append(name)

        '''
        the system as follows 
        Roll the layer lottery for the new node (same as before)
        Add it to the graph, no neighbors yet
        If it's the very first node, stop — nothing to connect to
        Otherwise, gather everyone ELSE already in the graph into a (name, vector) list
        Use your OWN search() (the cosine one from search.py) to find the k closest among them
        For each match, wire the connection BOTH ways
        
        
        '''


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


