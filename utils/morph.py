import spacy
from stop_words import get_stop_words
from traceback import format_exc
import sys


POS_SEP = "|"
GO_POS = ["NOUN", "VERB", "ADJ"]
STOP_LIST = [str(w) for w in
             ['.', ',', '?', '!', ";", ':', '"', "&", "'", "(", ")", "-", "+", "/", "\\", "|","[","]", "often", "a", "an",
              "-pron-","can", "just"]]


def load_stoplist():
    try:
        return set(get_stop_words("en") + STOP_LIST)
    except:
        print((format_exc()))
        return set()


# load resources 
_stop_words = load_stoplist()
print("Loading spacy model...")
_spacy = spacy.load('en')


def get_stoplist():
    return _stop_words


def lemmatize(text, lowercase=True, lang="en"):
    """ Return lemmatized text """
    
    tokens = _spacy(text, tag=True, parse=False, entity=True)
    text_lemmatized = " ".join(t.lemma_ for t in tokens)
    
    if lowercase:
        text_lemmatized = text_lemmatized.lower()

    return text_lemmatized


def add_pos(text, pos_sep=POS_SEP):
    """ Add POS tags to input text e.g. 'Car|NOUN is|VERB blue|ADJ.' """
    tokens = _spacy(text, tag=True, parse=False, entity=True)
    text_pos = " ".join(t.orth_ + pos_sep + t.pos_ for t in tokens)
    pos = " ".join([t.pos_ for t in tokens])
    return text_pos, pos


def tokenize(text, pos_filter=False, lowercase=True, remove_stopwords=True, return_pos=False):
    tokens = _spacy(text, tag=True, parse=False, entity=False)
    lemmas = [t.lemma_ for t in tokens if not pos_filter or t.pos_ in GO_POS]
    if remove_stopwords: lemmas = [l for l in lemmas if l not in _stop_words and l.lower() not in _stop_words]
    if lowercase: lemmas = [l.lower() for l in lemmas]

    if return_pos: 
        res = []
        for t in tokens: res.append((t.lemma_, t.pos_))
        return res
    else:
       
        return lemmas


def lemmatize_word(word, lowercase=True):
    try:
        if len(word) == 0: return word
        tokens = _spacy(word, tag=True, parse=False, entity=False)
        if len(tokens) == 0: return word
        lemma = tokens[0].lemma_
        if lowercase: lemma = lemma.lower()
        return lemma
    except SystemExit as KeyboardInterrupt:
         sys.exit()
    except:
        print(("Warning: lemmatization error '%s'" % word))
        print((format_exc()))
        return word


def analyze_word(word, lowercase=True):
    tokens = _spacy(word, tag=True, parse=False, entity=False)
    lemma = tokens[0].lemma_
    if lowercase: lemma = lemma.lower()
    return lemma, tokens[0].pos_


def parse(text, pos_filter=False, lowercase=True, remove_stopwords=False):
    tokens = _spacy(text, tag=True, parse=True, entity=False)
    lemmas = [t.lemma_ for t in tokens if not pos_filter or t.pos_ in GO_POS]
    if remove_stopwords: lemmas = [l for l in lemmas if l not in _stop_words]
    if lowercase: lemmas = [l.lower() for l in lemmas]
    return lemmas
