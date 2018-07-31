'''This module loads a pickle file built by context_token_modeler.py and tests it against 1000 lemmatized sentences of Latin'''

from os import listdir
from os.path import isfile, join, expanduser
from collections import defaultdict
from pprint import PrettyPrinter
from cltk.tokenize.word import WordTokenizer
from cltk.stem.latin.j_v import JVReplacer
from tesserae.utils import TessFile
from cltk.semantics.latin.lookup import Lemmata
from operator import itemgetter
from cltk.utils.file_operations import open_pickle
from difflib import SequenceMatcher
import pickle



def compare_context(target, context):
    '''Compares the number of shared words between a token's context and the contexts of its possible lemmas.'''
    lemmas = lemmatizer.lookup([target])
    lemmas = lemmatizer.isolate(lemmas)
    if len(lemmas) > 1:
        shared_context_counts = dict()
        for lem in lemmas:
            lemma_context_dictionary = SKIP_LIBRARY[lem]
            lemma_context_words = lemma_context_dictionary.keys()
            counts = [lemma_context_dictionary[context_token] for context_token in set(context).intersection(lemma_context_words)]
            shared_context_counts[lem] = sum(counts)
            #print(shared_context_counts[lem])
        total_shared = sum(shared_context_counts.values())
        lemmalist = []
        # if the word has not been seen during training, just return an even distribution of probability.
        if total_shared > 0:
            for lem in lemmas:
                lemmaprob = shared_context_counts[lem] / total_shared
                lemmaobj = (lem, lemmaprob)
                lemmalist.append(lemmaobj)
        else: 
            lemmalist = lemmatizer.lookup([target])
            lemmalist = lemmalist[0][1]
        return lemmalist
    else:
        lemmalist = []
        lemmaobj = (lemmas[0], 1)
        lemmalist.append(lemmaobj)
        return lemmalist

tessobj = TessFile(onlyfiles[258])
tokengenerator = iter(tessobj.read_tokens())
tokens = new_file(tokengenerator, 2)
target = tokens.pop(0)
compare_context(target, tokens)


rel_path = os.path.join('~/cltk_data/latin/model/latin_models_cltk/lemmata/backoff')
path = os.path.expanduser(rel_path)
file = 'latin_pos_lemmatized_sents.pickle'
latin_pos_lemmatized_sents_path = os.path.join(path, file)
if os.path.isfile(latin_pos_lemmatized_sents_path):
    latin_pos_lemmatized_sents = open_pickle(latin_pos_lemmatized_sents_path)
else:
    print('The file %s is not available in cltk_data' % file)
first1000 = latin_pos_lemmatized_sents[0:1000]
first1000tokens = []
for sentence in first1000:
    for tup in sentence:
        if 'punc' not in tup[1]:
            first1000tokens.append(tup[0])

first1000lemmas = []
for sentence in first1000:
    for tup in sentence:
        if 'punc' not in tup[1]:
            first1000lemmas.append(tup[1])

first10 = latin_pos_lemmatized_sents[0:10]
first10tokens = []
for sentence in first10:
    for tup in sentence:
        if 'punc' not in tup[1]:
            first10tokens.append(tup[0])

first10lemmas = []
for sentence in first10:
    for tup in sentence:
        if 'punc' not in tup[1]:
            first10lemmas.append(tup[1])



def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def test_lemmatization(test_tokens, answer_lemma, window_size):
    '''Takes a sequence of tokens from the corpus and lemmatizes them using compare_context.
    Then checks against a set of 'gold standard' lemmatizations. Returns the incorrect answers.'''
    current_token = 0
    results = []
    correct = 0
    trials = 0
    roughly_correct = 0
    while current_token < len(test_tokens):
        if current_token >= window_size and current_token <= (len(test_tokens) - window_size):
            #print('current_token: ' + current_token)
            context = test_tokens[(current_token - window_size):(current_token + window_size)]
            #print(context)
            target = context.pop(window_size)
            #print(target, context)
            lemmas = compare_context(target, context)
            lemma = max(lemmas,key=itemgetter(1))
            # make a list of tuples containing just the token and the 2 lemmas
            results.append((target, lemma[0], answer_lemma[current_token]))
            trials = trials + 1
            if similar(lemma[0], answer_lemma[current_token]) > .7:
                roughly_correct = roughly_correct + 1
            if lemma[0] == answer_lemma[current_token]:
                correct = correct + 1
        current_token = current_token + 1
    print(correct)
    print(roughly_correct)
    print(trials)
    #print (correct + ' correct lemmatizations for ' + trials + ' trials. ' + (correct / trials) + ' accuracy.')
    return results


smalltest = test_lemmatization(first10tokens, first10lemmas, 2)
bigtest = test_lemmatization(first1000tokens, first1000lemmas, 2)

def find_errors(testobj):
    errors = [testobj[i] for i in range(0, len(testobj)) if testobj[i][1] != testobj[i][2]]
    return errors
