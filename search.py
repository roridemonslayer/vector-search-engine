def gap_sum(v1, v2): #v1 is the query and v2. is the item the v1 and v2 stuff are just parameters
    total = 0  # running tally of all the gaps
    for i in range(len(v1)): # len = count (3), range = walker (0,1,2)
        gap = abs(v1[i] - v2[i]) # v1 and v2 are two querys with two vector lists So v1[i] - v2[i] is tiny: one number minus one number. 0.85 - 0.80 = 0.05. it's pulling it from wtv postion its at with i 
        total += gap
    return total 

def search(query,catalog, k=3):
    result = [] #the scores go here
    for item in catalog: # goes to each catalog entry
        name = item[0] # pull out the item name so item[0] is the slot name
        vector = item[1] # this puls out the vector slot
        gap = gap_sum(query, vector)
        result.append((name, gap))
    ranked = sorted(result, key = lambda pair:pair[1]) #this is ouput
    return ranked[:k]


query = [0.85, 0.9, 0.05]
catalog = [
    ("tee A" , [0.9,0.8, 0.1]), 
    ("tee b", [0.8,  0.9, 0.1]),
    ("gown", [0.1, 0.8, 0.9]),
]
print(search(query, catalog, k = 1))
