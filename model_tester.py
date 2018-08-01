'''This module loads a pickle file built by context_token_modeler.py and tests it against 1000 lemmatized sentences of Latin'''

from os import listdir
from os.path import isfile, join, expanduser
import os
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
from random import randint

SKIP_LIBRARY = pickle.load( open( "skip_library_1_no_sentence_boundaries.pickle", "rb" ) )

def compare_context(target, context, control = 0):
    '''Compares the number of shared words between a token's context and the contexts of its possible lemmas.'''
    lemmas = lemmatizer.lookup([target])
    lemmas = lemmatizer.isolate(lemmas)
    if len(lemmas) > 1:
        if control == 1:
            lemmalist = []
            choice = randint(0, (len(lemmas) - 1))
            lemmaobj = (lemmas[choice], 1)
            lemmalist.append(lemmaobj)
            return lemmalist
        #this will return an even distribution, which will result in the 1st result being picked.
        if control == 2:
            lemmalist = lemmatizer.lookup([target])
            lemmalist = lemmalist[0][1]
            return lemmalist        
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


def new_sentence(tokengenerator, context_window):
    '''Takes an iterator object for the sentence being read.
    Reads in the first context_window * 2 + 1 tokens and returns them'''
    tokens = []
    for i in range(0, (context_window + 1)):
        rawtoken = next(tokengenerator)
        tokens.append(rawtoken)
    return tokens


def test_lemmatization(annotated_lemmas, window_size, control = 0):
    '''Takes a sequence of tokens from the corpus and lemmatizes them using compare_context.
    Then checks against a set of 'gold standard' lemmatizations. Returns the incorrect answers.'''
    current_token = 0
    results = []
    correct = 0
    trials = 0
    roughly_correct = 0
    for sentence in annotated_lemmas:
        test_tokens = [tup[0] for tup in sentence if 'punc' not in tup[1]]
        answer_lemma = [tup[1] for tup in sentence if 'punc' not in tup[1]]
        #sentence_iterator = iter(test_tokens)
        #tokens = new_sentence(sentence_iterator, context_window)
        while current_token < len(test_tokens):
            if current_token >= window_size and current_token <= (len(test_tokens) - window_size):
                #print('current_token: ' + current_token)
                context = test_tokens[(current_token - window_size):(current_token + window_size)]
                #print(context)
                target = context.pop(window_size)
                #print(target, context)
                lemmas = compare_context(target, context, control)
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


smalltest = test_lemmatization(first10, 2, control = 1)
bigtest = test_lemmatization(first1000tokens, first1000lemmas, 2, control = 0)

def find_errors(testobj):
    errors = [testobj[i] for i in range(0, len(testobj)) if testobj[i][1] != testobj[i][2]]
    return errors
