import csv
from collections import Counter
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from itertools import product
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine

wn = WordNetCorpusReader("./WordNet-3.0/dict",None)

with open('./SimLex-999/SimLex-999.txt') as f:
    entries = { (entry['word1'],entry['word2']): entry
                for entry in csv.DictReader(f, delimiter='\t') if entry['POS'] == 'A'}

def similarity(a,b):
    "Get the similarity between words A and B."
    return float(entries[(a,b)]['SimLex999'])

def pair_to_synsets(a,b):
    "Get the first sense in WordNet for a and b."
    return wn.synset(a + '.a.01'), wn.synset(b + '.a.01')

def pair_to_all_synsets(a,b):
    "Get all synsets in WordNet for a and b."
    return wn.synsets(a, 'a'), wn.synsets(b, 'a')

def shortest_distance(l1,l2):
    "Get the shortest path distance between all the items in l1 and l2."
    return min(a.shortest_path_distance(b) for a,b in product(l1,l2))

def average_distance(l1,l2):
    "Get the average distance between the items in l1 and l2."
    distances = [a.shortest_path_distance(b) for a,b in product(l1,l2)]
    return float(sum(distances))/len(distances)

def has_attributes(a):
    "Return True iff the synset has attributes."
    return bool(wn.synset(a+'.a.01').attributes())

pairs = {(a,b) for a,b in entries if has_attributes(a) and has_attributes(b)}

def attribute_set(a):
    "Return the set of all attributes for a."
    return {attr for synset in wn.synsets(a,'a') for attr in synset.attributes()}

# Define five different types of distances between adjectives based on attributes.
# Sadly, these don't really work.

def attribute_distance_1(a,b):
    l1 = attribute_set(a)
    l2 = attribute_set(b)
    return shortest_distance(l1,l2)

def attribute_distance_2(a,b):
    l1 = attribute_set(a)
    l2 = attribute_set(b)
    return len(l1&l2)#/ (float(len(l1)+len(l2))/2)

def attribute_distance_3(a,b):
    l1 = wn.synset(a +'.a.01').attributes()
    l2 = wn.synset(b +'.a.01').attributes()
    return shortest_distance(l1,l2)

def attribute_distance_4(a,b):
    l1 = set(wn.synset(a +'.a.01').attributes())
    l2 = set(wn.synset(b +'.a.01').attributes())
    return len(l1&l2)

def attribute_distance_5(a,b):
    l1 = attribute_set(a)
    l2 = attribute_set(b)
    return average_distance(l1,l2)

def correlation(pairs, distance_function):
    similarity_list = []
    distance_list = []
    for a,b in pairs:
        similarity_list.append(similarity(a,b))
        distance_list.append(distance_function(a,b))
    return spearmanr(similarity_list, distance_list)
