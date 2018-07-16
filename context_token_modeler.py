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

SKIP_LIBRARY = dict()
'''Large dictionary whose keys are inflected word forms and whose values are dictionaries.
Second-layer dictionaries describe context of word forms in corpus through counts of 
surrounding word-forms.'''


def read_files(filepath, context_window):
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
    while stop != 1:
        #the target should be five away from the end of the file, until the end
        target_position = len(tokens) - (context_window + 1)
        targettoken = tokens[target_position]
        #grab all the other tokens but the target
        contexttokens = [x for i, x in enumerate(tokens) if i != target_position]
        #add this context to the skipgram map
        skipgram(targettoken, contexttokens)
        #prep the next token in the file
        try:
            rawtoken = next(tokengenerator)
            cleantoken = token_cleanup(rawtoken)            
            tokens.append(cleantoken)
            if len(tokens) > (context_window * 2 + 1):
                tokens.pop(0)
        except StopIteration:
            #we have reached EOF. Loop through until the last token is done then quit
            #when this happens, the token list should have 11 indices, and the 'target_position'
            #index will be the sixth (i.e. :tokens[5]). Pop the first index off, leaving 10
            #indices and making the sixth index (previously the seventh) the new target.
            while len(tokens) > (context_window):
                tokens.pop(0)
                # This loop makes the target_position move to the end. E.g. if the context_window is 6, then
                # as long as there are six or more indexes, make the target_position the sixth index.
                if len(tokens) > (context_window + 1):
                    target_position = (context_window)
                # But if there six or fewer indexes, then the target_position is the last index.
                else:
                    target_position = len(tokens) - 1
                targettoken = tokens[target_position]
                #grab all the other tokens but the target
                contexttokens = [x for i, x in enumerate(tokens) if i != target_position]
                #add this context to the skipgram map
                skipgram(targettoken, contexttokens)
            stop = 1


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
    if targettoken not in SKIP_LIBRARY:
        SKIP_LIBRARY[targettoken] = defaultdict(int)
    for contextword in contexttokens:
        SKIP_LIBRARY[targettoken][contextword] += 1

def new_file(tokengenerator, context_window):
    '''Takes an iterator object for the file being read.
    Reads in the first six tokens and returns them'''
    tokens = []
    for i in range(0, (context_window + 1)):
        rawtoken = next(tokengenerator)
        cleantoken = token_cleanup(rawtoken)
        # NB: right now the code assumes that first sentence is > 5 words
        tokens.append(cleantoken)
    return tokens

def token_cleanup(rawtoken):
    jv = JVReplacer()
    word_tokenizer = WordTokenizer('latin')
    rawtoken = jv.replace(rawtoken)
    rawtoken = rawtoken.lower()
    tokenlist = word_tokenizer.tokenize(rawtoken)
    #while len(tokenlist) > 1:
        #sometimes the tokenizer finds punctuation.
        #when this happens, make sure all consecutive punctuation is gone.
        #NB: performs the same on quotes and all other punct, including commas.
     #   tokenlist = word_tokenizer.tokenize(tokenlist[0])
    return tokenlist[0]

word_tokenizer = WordTokenizer('latin')
pp = PrettyPrinter(indent=4)

#open all the tesserae files
relativepath = join('~/cleantess/tesserae/texts/la')
path = expanduser(relativepath)
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
onlyfiles = [join(path, f) for f in onlyfiles]

for filename in onlyfiles:
    print(filename)
    if '.tess' in filename:
        read_files(filename, context_window = 2)


'''With the context token counts complete, it's time to start associating lemmas 
with their summary contextual dictionaries. That means finding all the possible 
reflexes of a lemma, and then looking them up in SKIP_DICTIONARY.'''

# find the inflected_forms on the left (keys), and associate them with the lemmas on the right (value lists)



[k for k, lemma_list in SKIP_LIBRARY.items() if  in ]