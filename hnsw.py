from search import cosine

graph = {
    "1": {"vector": [1, 8], "neighbors": ["2", "4"]},
    "2": {"vector": [3, 7], "neighbors": ["1", "6", "3"]},
    "3": {"vector": [4, 8], "neighbors": ["2"]},
    "4": {"vector": [0, 3], "neighbors": ["1"]},
    "5": {"vector": [7, 1], "neighbors": ["6"]},
    "6": {"vector": [8, 2], "neighbors": ["2", "5"]},
}

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

query = [7,1]      # sits right on top of node 3's vector
print(greedy_search(graph, query, start="1"))






