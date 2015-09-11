from collections import defaultdict
from scipy.stats import spearmanr
import csv

with open('./resources/SimLex-999/SimLex-999.txt') as f:
    entries = { (entry['word1'],entry['word2']): entry
                for entry in csv.DictReader(f, delimiter='\t') if entry['POS'] == 'A'}

def similarity(a,b):
    "Get the similarity between words A and B."
    return float(entries[(a,b)]['SimLex999'])

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

def hso_wordforms(selection_function=min):
    data = wordform_dict('./resources/pedersen_similarity/hso_result.txt')
    return {tuple: selection_function(value) for tuple, value in data.items()}

def min_avoiding_zero(l):
    s = set(l)
    if len(s) == 1:
        return l[0]
    else:
        return min(s-{0})

def evaluate():
    lesk_dict = lesk_wordforms()
    hso_dict1 = hso_wordforms(min)
    hso_dict2 = hso_wordforms(max)
    hso_dict3 = hso_wordforms(min_avoiding_zero)
    simlist   = []
    lesk_list = []
    hso1_list = []
    hso2_list = []
    hso3_list = []
    for a,b in entries:
        simlist.append(similarity(a,b))
        lesk_list.append(lesk_dict[(a,b)])
        hso1_list.append(hso_dict1[(a,b)])
        hso2_list.append(hso_dict2[(a,b)])
        hso3_list.append(hso_dict3[(a,b)])
    print spearmanr(lesk_list, simlist)
    print spearmanr(hso1_list, simlist)
    print spearmanr(hso2_list, simlist)
    print spearmanr(hso3_list, simlist)

def evaluate_limited():
    hso_dict1 = hso_wordforms(min_avoiding_zero)
    hso_dict2 = hso_wordforms(max)
    simlist   = []
    hso_list1 = []
    hso_list2 = []
    for pair, value in hso_dict2.items():
        if value == 0:
            continue
        simlist.append(similarity(*pair))
        hso_list1.append(hso_dict1[pair])
        hso_list2.append(hso_dict2[pair])
    print spearmanr(hso_list1, simlist)
    print spearmanr(hso_list2, simlist)
    print len(simlist)
