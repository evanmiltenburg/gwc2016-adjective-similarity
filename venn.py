from nltk.corpus.reader.wordnet import WordNetCorpusReader
from matplotlib import pyplot as plt
from matplotlib_venn import venn3_unweighted

wn = WordNetCorpusReader("./resources/WordNet-3.0/dict",None)

adjectives = {a for a in wn.all_synsets('a')}
attributes = {n for n in wn.all_synsets('n') if n.lexname() == 'noun.attribute'}

direct_attributes = {attribute for adjective in adjectives
                               for attribute in adjective.attributes()}
morphologically_related = {related_lemma.synset() for adjective in adjectives
                                                  for lemma in adjective.lemmas()
                                                  for related_lemma in lemma.derivationally_related_forms()
                                                  if related_lemma.synset().pos() == 'n'}

diagram = venn3_unweighted([attributes, direct_attributes, morphologically_related],
                ['labeled as\nnoun.attribute', 'direct\nattributes', 'morphologically\nrelated nouns'])

for patch in diagram.patches:
    patch.set_edgecolor('k')
    patch.set_facecolor('w') # remove this line for color diagram.

plt.savefig('./images/venn.pdf')
