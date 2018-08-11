'''This module is designed to scan the entire Tesserae text folder and build contextual data
for each type in the corpus. This is necessary for certain experiments involving the ideal
lemma (in ambiguous cases), synonym, or translation for a given token-in-context during
NLP tasks. Contributors: James Gawley.
'''

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






'''Large dictionary whose keys are inflected word forms and whose values are dictionaries.
Second-layer dictionaries describe context of word forms in corpus through counts of 
surrounding word-forms.'''
punctuation_list = ['.', '!', ';', ':', '?']

def read_files_skipgram(filepath, context_window):
    '''Moves through a .tess file and calls the 'next' and 'skipgram' functions as needed.
    Updates the SKIP_LIBRARY global object.
    Parameters
    ----------
    filepath: a file in .tess format
    context_window: how many words on either side of the target to look at.
    '''
    tessobj = TessFile(filepath)
    tokengenerator = iter(tessobj.read_tokens())
    tokens = new_file(tokengenerator, context_window)
    stop = 0
    clearflag = 0
    target_position = context_window
    while stop != 1:
        #the target should be five away from the end of the file, until the end
        # can't just pop the target token; we want to keep it for the next round.
        targettoken = tokens[target_position]
        #grab all the other tokens but the target
        contexttokens = [x for i, x in enumerate(tokens) if i != target_position]
        #add this context to the skipgram map
        skipgram(targettoken, contexttokens)
        #prep the next token in the file
        try:
            rawtoken = next(tokengenerator)
            cleantoken_list = token_cleanup(rawtoken) 
            if len(cleantoken_list) > 1 and cleantoken_list[-1] in punctuation_list:
                #this should indicate a sentence has ended.
                #when this happens, it's necessary to clear the list *after* this iteration.
                clearflag = 1
            tokens.append(cleantoken_list[0])
            # if we've seen end-of-sentence punctuation, we need to start counting down.
            if clearflag == 1:
                # when this begins, the token list just received the final word.
                tokens.pop(0)
                while len(tokens) > context_window:
                    # perform the usual dictionary operation, but don't add a new token.
                    targettoken = tokens[target_position]
                    contexttokens = [x for i, x in enumerate(tokens) if i != target_position]
                    skipgram(targettoken, contexttokens)
                    tokens.pop(0)
                #initialize the next sentence
                tokens = []
                tokens = new_file(tokengenerator, context_window)
                clearflag = 0
            else:
                tokens.pop(0)
        except StopIteration:
            #we have reached EOF. Loop through until the last token is done then quit
            #when this happens, the token list should have 11 indices, and the 'target_position'
            #index will be the sixth (i.e. :tokens[5]). Pop the first index off, leaving 10
            #indices and making the sixth index (previously the seventh) the new target.
            # this entire loop is obsolete now that punctuation is accounted for.
            try:
                tokens.pop(0)
            except IndexError:
                pass
            while len(tokens) > (context_window):
                # This loop makes the target_position move to the end. E.g. if the context_window is 6, then
                # as long as there are six or more indexes, make the target_position the sixth index.
                targettoken = tokens[target_position]
                #grab all the other tokens but the target
                contexttokens = [x for i, x in enumerate(tokens) if i != target_position]
                #add this context to the skipgram map
                skipgram(targettoken, contexttokens)
                tokens.pop(0)
            stop = 1


def read_files_count(filepath):
    tessobj = TessFile(filepath)
    tokengenerator = iter(tessobj.read_tokens())
    stop = 0
    while stop != 1:
        try:
            rawtoken = next(tokengenerator)
            cleantoken_list = token_cleanup(rawtoken) 
            token = cleantoken_list[0]
            countgram(token)
        except StopIteration:
            stop = 1

def countgram(targettoken):
    global COUNT_LIBRARY
    lemmas = lemmatizer.lookup([targettoken])
    lemmas = lemmatizer.isolate(lemmas)
    if len(lemmas) == 1:
        COUNT_LIBRARY[lemmas[0]] += 1

lemmatizer = Lemmata(dictionary = 'lemmata', language = 'latin')
def skipgram(targettoken, contexttokens):
    '''Builds a complex data structure that will contain the 'average context'
    for each type in the corpus.
    param targettoken: the token in question
    param c: the context tokens
    global SKIP_LIBRARY: a dictionary whose keys are types and whose values are
    dictionaries; in turn their keys are context types and values are
    incremented counts.
    '''
    global SKIP_LIBRARY
    lemmas = lemmatizer.lookup([targettoken])
    lemmas = lemmatizer.isolate(lemmas)
    for lemma in lemmas:
        if lemma not in SKIP_LIBRARY:
            SKIP_LIBRARY[lemma] = defaultdict(int)
        for contextword in contexttokens:
            SKIP_LIBRARY[lemma][contextword] += 1
            SKIP_LIBRARY[lemma]['times_seen'] += 1

def new_file(tokengenerator, context_window):
    '''Takes an iterator object for the file being read.
    Reads in the first context_window * 2 + 1 tokens and returns them'''
    tokens = []
    for i in range(0, (context_window + 1)):
        rawtoken = next(tokengenerator)
        cleantoken_list = token_cleanup(rawtoken)
        # NB: right now the code assumes that first sentence is < 2x the window + 1
        tokens.append(cleantoken_list[0])
    return tokens

jv = JVReplacer()
word_tokenizer = WordTokenizer('latin')
def token_cleanup(rawtoken):
    # this cleaning algorithm is a potential area for improvement.
    rawtoken = jv.replace(rawtoken)
    rawtoken = rawtoken.lower()
    tokenlist = word_tokenizer.tokenize(rawtoken)
    #sometimes words are split into enclitics and punctuation.
    return tokenlist

word_tokenizer = WordTokenizer('latin')
pp = PrettyPrinter(indent=4)

#open all the tesserae files
relativepath = join('~/cleantess/tesserae/texts/la')
path = expanduser(relativepath)
onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and 'augustine' not in f and 'ambrose' not in f and 'jerome' not in f and 'tertullian' not in f and 'eugippius' not in f and 'hilary' not in f]
onlyfiles = [join(path, f) for f in onlyfiles]

SKIP_LIBRARY = dict()
COUNT_LIBRARY = defaultdict(int)
for filename in onlyfiles:
    print(filename)
    if '.tess' in filename:
        read_files_skipgram(filename, context_window = 2)
        read_files_count(filename)

relativepath = join('~/latin_lemma_disambiguation_models')
path = expanduser(relativepath)
pickle_file = join(path, "skip_library_1_no_sentence_boundaries.pickle")
pickle.dump( SKIP_LIBRARY, open( pickle_file, "wb" ) )