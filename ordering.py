import matplotlib
from tabulate import tabulate
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import csv
from collections import defaultdict
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from itertools import product
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
from math import floor

sns.set_style("whitegrid")

################################################################################
# Evaluation function by Le & Fokkens (2015)

def score_with_tie_correction(ab_pred, xy_pred, ab_gold, xy_gold):
    '''
    Score a similarity measure against a gold standard on the comparison of
    pairs of pairs (a,b) and (x,y)
    '''
    g = np.sign(ab_gold-xy_gold)
    p = np.sign(ab_pred-xy_pred)
    if g == 0 or p == 0: return 0.5
    if g == p: return 1
    return 0

def accuracy(gold, predicted, score_func=score_with_tie_correction):
    '''
    Accept two lists of similarity scores and output ordering accuracy.
    There are two possible scoring functions, see "score_naive" and
    "score_with_tie_correction".
    '''
    n = len(gold)
    assert n == len(predicted)
    correct = 0
    count = 0
    for i in range(n):
        for j in range(n):
            if predicted[i] is None or predicted[j] is None: continue
            correct += score_func(predicted[i], predicted[j], gold[i], gold[j])
            count += 1
    return float(correct)/count, count

def accuracy_by_group(gold, predicted, group_indices,
                      score_func=score_with_tie_correction):
    '''
    Accept two lists of similarity scores and a list of group indices.
    Returns two elements:
    - a Numpy matrix of component scores, cell [i,j] corresponds to pairs with
      one element in group i and the other in group j
    - a Numpy matrix of component weights, i.e. the number of comparisons
      corresponding to each component score
    '''
    groups = defaultdict(list)
    for i, g, s in zip(group_indices, gold, predicted):
        if s is None: continue
        groups[i].append((g, s))
    n = max(groups.keys())+1
    correct = np.zeros((n, n), dtype=float)
    count = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            count[i,j] = len(groups[i])*len(groups[j])
            for ab in groups[i]:
                for xy in groups[j]:
                    correct[i,j] += score_func(ab[1], xy[1], ab[0], xy[0])
    return correct/count, count

def accuracy_by_group(gold, predicted, group_indices,
                      score_func=score_with_tie_correction):
    '''
    Accept two lists of similarity scores and a list of group indices.
    Returns two elements:
    - a Numpy matrix of component scores, cell [i,j] corresponds to pairs with
      one element in group i and the other in group j
    - a Numpy matrix of component weights, i.e. the number of comparisons
      corresponding to each component score
    '''
    groups = defaultdict(list)
    for i, g, s in zip(group_indices, gold, predicted):
        if s is None: continue
        groups[i].append((g, s))
    n = max(groups.keys())+1
    correct = np.zeros((n, n), dtype=float)
    count = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            count[i,j] = len(groups[i])*len(groups[j])
            for ab in groups[i]:
                for xy in groups[j]:
                    correct[i,j] += score_func(ab[1], xy[1], ab[0], xy[0])
    return correct/count, count

# Modified from Le and Fokkens (2015):

def evaluate_groups(gold, predicted, score_func=score_with_tie_correction, group_num=5):
    group_indices = [int(floor(g*group_num/10)) for g in gold]
    a, c = accuracy_by_group(gold, predicted, group_indices, score_func)
    results = []
    print("\tDelta\tAccuracy\t#Pairs")
    for k in range(group_num):
        correct = 0.0
        count = 0
        for i in range(group_num):
            for j in range(group_num):
                if abs(i-j) == k:
                    correct += a[i,j]*c[i,j]
                    count += c[i,j]
        accuracy = correct / count
        results.append((k,accuracy))
        print("\t%d\t%f\t%d" %(k, accuracy, count))
    return results

# Own functions:

def results_to_table(res1,res2,name1,name2):
    d1 = dict(res1)
    d2 = dict(res2)
    def make_percent(v):
        return int(v*100)
    header = ["Delta",name1,name2]
    table = [[k, make_percent(d1[k]), make_percent(d2[k])] for k in d1]
    print(tabulate(table, headers=header, tablefmt="latex_booktabs"))

def results_to_df(results1, results2, method1, method2):
    delta1, score1 = map(list,zip(*results1))
    delta2, score2 = map(list,zip(*results2))
    return pd.DataFrame({
            'Method': ([method1] * len(results1)) + ([method2] * len(results2)),
            'Difference between pairs': delta1 + delta2,
            'Percentage correct': list(map(lambda x: int(100*x),score1 + score2)),
            })

def plot(df,filename='./images/plot.pdf'):
    ax = sns.barplot(x='Difference between pairs',
                     y="Percentage correct",
                     hue="Method",
                     data=df)
    ax.set_xlabel('Difference between pairs')
    ax.set_ylabel('Percentage correct')
    plt.savefig(filename)
################################################################################

# Load WordNet:
wn = WordNetCorpusReader("./resources/WordNet-3.0/dict",None)

# Load SimLex-999 data:
with open('./resources/SimLex-999/SimLex-999.txt') as f:
    entries = { (entry['word1'],entry['word2']): entry
                for entry in csv.DictReader(f, delimiter='\t') if entry['POS'] == 'A'}

def similarity(a,b):
    "Get the similarity between words A and B."
    return float(entries[(a,b)]['SimLex999'])

# Load vector data:
adjs = {w for pair in entries for w in pair}
with open('./resources/Predict-vector/EN-wform.w.5.cbow.neg10.400.subsmpl.txt') as f:
    vectors = dict()
    for line in f:
        row = line.split('\t')
        if row[0] in adjs:
            vectors[row[0]] = list(map(float,row[1:]))

################################################################################
# Load lesk similarity data:

def get_wnsim_data(filename):
    with open(filename) as f:
        data = []
        for line in f:
            if line == '\n':
                continue
            elif line[-2].isdigit():
                data.append(line.strip().split())
        return {tuple(entry[:2]):int(entry[2]) for entry in data}

def wordform_dict(filename):
    data = get_wnsim_data(filename)
    results = defaultdict(list)
    for a,b in data:
        results[(a.split('#')[0],b.split('#')[0])].append(data[(a,b)])
    return results

def lesk_wordforms():
    data = wordform_dict('./resources/pedersen_similarity/lesk_result.txt')
    return {tuple: max(value) for tuple, value in data.items()}

################################################################################
# Define functions to get distance based on derivationally related word forms

def get_related_nouns(synset):
    "For a given synset, get all the derivationally related noun synsets."
    return [related.synset() for lemma in synset.lemmas()
                             for related in lemma.derivationally_related_forms()
                             if related.synset().pos() == 'n']

def pair_to_synsets(a,b):
    "Get the first sense in wordnet for two adjectives."
    return wn.synset(a + '.a.01'), wn.synset(b + '.a.01')

def have_related_forms(a,b):
    """
    Check if both a and b have related forms.
    A word has related forms iff the first synset has related nouns.
    """
    a,b = pair_to_synsets(a,b)
    return bool(get_related_nouns(a)) and bool(get_related_nouns(b))

def shortest_distance(l1,l2):
    "Return the shortest distance between two lists of synsets."
    return min(a.shortest_path_distance(b) for a,b in product(l1,l2))

def related_distance(a,b):
    "Return the distance between A and B in terms of their related word forms."
    l1  = [related for synset in wn.synsets(a,'a') for related in get_related_nouns(synset)]
    l2  = [related for synset in wn.synsets(b,'a') for related in get_related_nouns(synset)]
    return shortest_distance(l1,l2)

################################################################################
# Functions for the hybrid approach:

def ranking_for_list(l):
    """Takes list of scores, returns a list of positions and their ranks.
    
    Ex: ranking_for_list([1,3,2]) returns [(0, 0), (1, 2), (2, 1)]"""
    ranking = [item for item,score in sorted(enumerate(l), key=lambda t:t[1])]
    return list((item,rank) for rank,item in enumerate(ranking))

def order_list(preferred, secondary):
    """
    This is the main function that takes two lists of scores and returns a dictionary
    with the original positions as keys and rankings as values.
    """
    # make sure all is OK:
    assert len(preferred) <= len(secondary)
    
    # Create index for the primary ordering (the one we'll return):
    order         = dict(ranking_for_list(preferred))
    # If both lists are equally long (no missing items from the preferred ordering)
    # keep the preferred ordering.
    if len(preferred) == len(secondary):
        return order
    
    # create indices for the secondary order:
    order2        = dict(ranking_for_list(secondary))
    reverse2      = dict((pos,item) for item,pos in order2.items())
    
    # set the missing items apart and order them among themselves:
    to_be_ordered = list(range(len(preferred),len(secondary)))
    pre_order     = sorted(to_be_ordered, key=lambda item: order2[item])
    
    # special operation for the first item: if it should actually be the first item,
    # we need to add it to the front of the ordering.
    current = pre_order[0]
    if order2[current] == 0:
        for key in order:
            order[key] += 1
        order[current] = 0
        pre_order = pre_order[1:]
    # main loop:
    for item in pre_order:
        position            = order2[item]
        before              = reverse2[position-1]
        maximal_position    = max(order.values())
        # If we need to put the item at the end of the list:
        if order[before] == maximal_position:
            order[item] = maximal_position + 1
            continue
        # If we need to make place in the list, items with lower similarity move
        # one step to the right, and we insert the item in the right place.
        for key, value in order.items():
            if value > order[before]:
                order[key] += 1
        order[item] = order[before] + 1
    return order

def rank_scores(l1,l2):
    "This function simply reduces the dictionary to a list of ranks."
    return [b for a,b in sorted(order_list(l1,l2).items())]

def tiebreaker(preferred,secondary,bool_reverse=False):
    "Use the second list to break ties in the first list."
    orderdict = dict()
    posscore = dict(enumerate(preferred))
    posscore_vectors = dict(enumerate(secondary))
    scorepos = defaultdict(list)
    for pos, score in posscore.items():
        scorepos[score].append(pos)
    start = 0
    for score in sorted(scorepos.keys()):
        max_index   = start + len(scorepos[score])
        new_indices = range(start,max_index)
        # sort all the positions with the same WordNet-score using the cosine similarity.
        # sort from most to least similar (increasing cosine distance).
        new_order   = sorted(scorepos[score],
                             key=lambda pos: posscore_vectors[pos],
                             reverse=bool_reverse)
        orderdict.update(zip(new_order,new_indices))
        start = max_index
    return [b for a,b in sorted(orderdict.items())]


def order_list1(preferred, secondary):
    "Ordering with tie-breaker."
    preferred_noties = tiebreaker(preferred,secondary)
    return rank_scores(preferred_noties, secondary)

def order_list2(preferred, secondary):
    "Ordering without tie-breaker."
    posscore   = dict(enumerate(preferred))
    additional = list(range(len(preferred),len(secondary)))
    order_secondary = dict(ranking_for_list(secondary))
    reverse_order_secondary = dict((rank,pos) for pos,rank in order_secondary.items())
    additional = sorted(additional,key=lambda pos: order_secondary[pos])
    for pos in additional:
        rank = order_secondary[pos]
        if rank == 0:
            posscore[pos] = 0
        else:
            before        = reverse_order_secondary[rank-1]
            posscore[pos] = posscore[before]
    return [rank for pos, rank in sorted(posscore.items())]


################################################################################
# Statistics for the hybrid approach and the vector-based approach:

lesk_dict = lesk_wordforms()

# Define two sets of pairs: those with related forms in WordNet and those without.
in_wordnet      = []
not_in_wordnet  = []
for a,b in entries:
    if have_related_forms(a,b):
        in_wordnet.append((a,b))
    else:
        not_in_wordnet.append((a,b))

# Make lists to hold the scores.
wordnet_list    = []
vectors_list    = []
similarity_list = []
lesk_list       = []

# Generate the lists:
for a,b in in_wordnet:
    wordnet_list.append(related_distance(a,b))
    similarity_list.append(similarity(a,b))
    vectors_list.append(cosine(vectors[a],vectors[b]))
    lesk_list.append(lesk_dict[(a,b)])

# NOTE: accuracy has the gold standard as its FIRST argument!

def inverse(l):
    """
    Le and Fokkens defined their ordering accuracy such that the correlation is
    expected to be positive. If the correlation is negative, I use this function to
    invert the list.
    """
    maximal = max(l) + 0.1
    return list(map(lambda x: maximal-x, l))

corr = str(spearmanr(wordnet_list,similarity_list)[0])
acc  = str(accuracy(similarity_list, inverse(wordnet_list)))
print('Results for WordNet alone (subset): ' + corr)
print('Results with ordering accuracy:     ' + acc + '\n')
wordnet_results_subset = evaluate_groups(similarity_list,inverse(wordnet_list))

corr = str(spearmanr(vectors_list,similarity_list)[0])
acc  = str(accuracy(similarity_list,inverse(vectors_list)))
print('Results for the vectors alone (subset): ' + corr)
print('Results with ordering accuracy:         ' + acc + '\n')
vector_results_subset = evaluate_groups(similarity_list,inverse(vectors_list))

plt.clf()
df = pd.DataFrame({'WordNet': wordnet_list, 'Vectors': vectors_list})
g = sns.lmplot(x="WordNet", y="Vectors", data=df)#, markers=None)
plt.savefig('./images/regression_wordnet_vectors_subset.pdf')

################################################################################
# Plot intermediate results:

plt.clf()
df = results_to_df(wordnet_results_subset,vector_results_subset,'WordNet','Vectors')
plot(df,'./images/wordnet_vectors_subset.pdf')

results_to_table(wordnet_results_subset,vector_results_subset, 'WordNet', 'Vectors')



corr = str(spearmanr(lesk_list,similarity_list)[0])
acc  = str(accuracy(similarity_list,lesk_list))
print('Results for lesk alone (subset): ' + corr)
print('Results with ordering accuracy:  ' + acc + '\n')

# Continue filling the lists with words that don't have DR-wordforms in wordnet.
for a,b in not_in_wordnet:
    similarity_list.append(similarity(a,b))
    vectors_list.append(cosine(vectors[a],vectors[b]))
    lesk_list.append(lesk_dict[(a,b)])

corr = str(spearmanr(vectors_list,similarity_list)[0])
acc  = str(accuracy(similarity_list,inverse(vectors_list)))
print('Results for the vectors alone: ' + corr)
print('Results with ordering accuracy:  ' + acc + '\n')
vector_results = evaluate_groups(similarity_list,inverse(vectors_list))

corr = str(spearmanr(lesk_list,similarity_list)[0])
acc  = str(accuracy(similarity_list,lesk_list))
print('Results for lesk alone:         ' + corr)
print('Results with ordering accuracy: ' + acc + '\n')

# Create an ordered list using the hybrid approach
# Problem: ties make the results unreliable.

# hybrid = rank_scores(wordnet_list, vectors_list)
# corr = str(spearmanr(hybrid,similarity_list)[0])
# print('Results for the hybrid approach (vectors): ' + corr)

def stats(preferred,secondary):
    def median(l):
        s = sorted(l)
        n,remainder = divmod(len(l),2)
        if remainder == 1:
            return s[remainder+1]
        else:
            return (s[remainder] + s[remainder+1])/2
    pos_by_pref_score = defaultdict(list)
    for pos, score in enumerate(preferred):
        pos_by_pref_score[score].append(pos)
    vector_avg_by_pref_score = {score:median([secondary[pos] for pos in l])
                                for score,l in pos_by_pref_score.items()}
    print(vector_avg_by_pref_score.items())

stats(wordnet_list,vectors_list)

hybrid = order_list1(wordnet_list, vectors_list)
corr = str(spearmanr(hybrid,similarity_list)[0])
acc  = str(accuracy(similarity_list,inverse(hybrid)))
print('Results for the hybrid approach (vectors, no ties): ' + corr)
print('Results with ordering accuracy: ' + acc + '\n')
hybrid_results = evaluate_groups(similarity_list,inverse(hybrid))

plt.clf()
df = results_to_df(hybrid_results,vector_results,'Hybrid','Vectors')
plot(df,'./images/hybrid_vectors.pdf')

results_to_table(hybrid_results,vector_results, 'Hybrid', 'Vectors')

plt.clf()
df = pd.DataFrame({'Hybrid': hybrid, 'Vectors': vectors_list})
g = sns.lmplot(x="Hybrid", y="Vectors", data=df)#, markers=None)
plt.savefig('./images/regression_hybrid_vectors.pdf')

hybrid = order_list2(wordnet_list, vectors_list)
corr = str(spearmanr(hybrid,similarity_list)[0])
acc  = str(accuracy(similarity_list,inverse(hybrid)))
print('Results for the hybrid approach (vectors, with ties): ' + corr)
print('Results with ordering accuracy: ' + acc + '\n')

hybrid = rank_scores(wordnet_list, lesk_list)
corr = str(spearmanr(hybrid,similarity_list)[0])
print('Results for the hybrid approach (lesk): ' + corr)

wordnet_noties = tiebreaker(wordnet_list,lesk_list,bool_reverse=True)
hybrid = rank_scores(wordnet_noties, lesk_list)
corr = str(spearmanr(hybrid,similarity_list)[0])
acc  = str(accuracy(similarity_list,hybrid))
print('Results for the hybrid approach (lesk, no ties): ' + corr)
print('Results with ordering accuracy: ' + acc + '\n')
