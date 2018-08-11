'''This module is designed to scan the entire Tesserae text folder and build frequency data
for each possible lemma in the corpus. Contributors: James Gawley.
'''

from os import listdir
from os.path import isfile, join, expanduser
from cltk.tokenize.word import WordTokenizer
from cltk.stem.latin.j_v import JVReplacer
from tesserae.utils import TessFile
from cltk.semantics.latin.lookup import Lemmata
from operator import itemgetter
from cltk.utils.file_operations import open_pickle
import pickle

def read_files(filepath):
    '''Moves through a .tess file and calls the 'next' and 'count_lemma' functions as needed.
    Updates the SKIP_LIBRARY global object.
    Parameters
    ----------
    filepath: a file in .tess format
    '''
    tessobj = TessFile(filepath)
    tokengenerator = iter(tessobj.read_tokens())
    stop = 0
    while stop != 1:
        try: 
            rawtoken = next(tokengenerator)
            cleantoken_list = token_cleanup(rawtoken)
            count_lemma(cleantoken_list[0])
        except StopIteration:
            stop = 1


lemmatizer = Lemmata(dictionary = 'lemmata', language = 'latin')
def count_lemma(targettoken):
    '''Builds a complex data structure that will contain the 'average context'
    for each type in the corpus.
    param targettoken: the token in question
    param c: the context tokens
    global SKIP_LIBRARY: a dictionary whose keys are types and whose values are
    dictionaries; in turn their keys are context types and values are
    incremented counts.
    '''
    global COUNT_LIBRARY
    lemmas = lemmatizer.lookup([targettoken])
    lemmas = lemmatizer.isolate(lemmas)
    for lemma in lemmas:
        if lemma not in COUNT_LIBRARY:
            COUNT_LIBRARY[lemma] = 0
        COUNT_LIBRARY[lemma] = COUNT_LIBRARY[lemma] + 1

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

#open all the tesserae files
relativepath = join('~/cleantess/tesserae/texts/la')
path = expanduser(relativepath)
onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and 'augustine' not in f and 'ambrose' not in f and 'jerome' not in f and 'tertullian' not in f and 'eugippius' not in f and 'hilary' not in f]
onlyfiles = [join(path, f) for f in onlyfiles]


COUNT_LIBRARY = dict()
for filename in onlyfiles:
    print(filename)
    if '.tess' in filename:
        read_files(filename)

relativepath = join('~/latin_lemma_disambiguation_models')
path = expanduser(relativepath)
pickle_file = join(path, "skip_library_1_no_sentence_boundaries.pickle")
pickle.dump( SKIP_LIBRARY, open( pickle_file, "wb" ) )