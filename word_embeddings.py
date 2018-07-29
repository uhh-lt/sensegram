import gzip
import codecs
from gensim.utils import tokenize
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from time import time
from os.path import exists

class GzippedCorpusStreamer(object):
    def __init__(self, corpus_fpath):
        self._corpus_fpath = corpus_fpath
        
    def __iter__(self):
        if self._corpus_fpath.endswith(".gz"):
            corpus = gzip.open(self._corpus_fpath, "r", "utf-8")
        else:
            corpus = codecs.open(self._corpus_fpath, "r", "utf-8")
            
        for line in corpus:
                yield list(tokenize(line,
                              lowercase=False,
                              deacc=False,
                              encoding='utf8',
                              errors='strict',
                              to_lower=False,
                              lower=False))

def load_vocabulary(vocabulary_fpath):
    voc = set()
    with codecs.open(vocabulary_fpath, "r", "utf-8") as voc_file:
        for line in voc_file:
            word = line.strip()
            voc.add(word)
            voc.add(word.capitalize())
            word = word.replace(" ", "_")
            voc.add(word)
            voc.add(word.capitalize())

    return voc


def add_phrases(tokens, phrases, do_restore_bigrams):
    """ Add multiword phrases to the input sequence of tokens. """

    def get_ngram_max(phrases):
        max_len = 0

        for p in phrases:
            p_len = len(p.split("_"))
            if max_len < p_len: max_len = p_len

        return max_len

    def split_tokens(tokens):
        splitted_tokens = []

        for t in tokens:
            if "_" in t:
                splitted = t.split("_")
                for st in splitted:
                    splitted_tokens.append(st)
            else:
                splitted_tokens.append(t)

        return splitted_tokens

    def add_dict_phrases(tokens, phrases):
        splitted_tokens = split_tokens(tokens)
        ngram_max = get_ngram_max(phrases)

        phrase_tokens = []
        skip_tokens = 0

        for i in range(len(splitted_tokens)):
            if skip_tokens > 0:
                skip_tokens -= 1
                continue

            for ngram_size in range(ngram_max, 2 -1 , -1):
                phrase_candidate = "_".join(splitted_tokens[i:i + ngram_size])
                if phrase_candidate in phrases:
                    phrase_tokens.append(phrase_candidate)
                    print("+++", phrase_candidate)
                    skip_tokens = ngram_size - 1
                    break

            if skip_tokens == 0:
                phrase_tokens.append(splitted_tokens[i])

        return phrase_tokens

    def get_bigrams(tokens):
        bigrams = set()

        for t in tokens:
            if "_" in t: bigrams.add(t)

        return bigrams

    def restore_bigrams(tokens_with_phrases, tokens_with_bigrams):
        bigrams = get_bigrams(tokens_with_bigrams)

        tokens_with_phrases_and_bigrams = []
        skip = False
        for i in range(len(tokens_with_phrases)):
            if skip:
                skip = False
                continue

            bigram_candidate_space = " ".join(tokens_with_phrases[i:i+2])
            bigram_candidate_under = "_".join(tokens_with_phrases[i:i+2])

            if "_" not in bigram_candidate_space and bigram_candidate_under in bigrams:
                tokens_with_phrases_and_bigrams.append(bigram_candidate_under)
                skip = True
            else:
                tokens_with_phrases_and_bigrams.append(tokens_with_phrases[i])

        return tokens_with_phrases_and_bigrams

    tokens_with_phrases = add_dict_phrases(tokens, phrases)

    if do_restore_bigrams:
        return restore_bigrams(tokens_with_phrases, tokens)
    else:
        return tokens_with_phrases


def learn_word_embeddings(corpus_fpath, vectors_fpath, cbow, window, iter_num, size, threads,
                          min_count, detect_bigrams=True, phrases_fpath=""):

    tic = time()
    sentences = GzippedCorpusStreamer(corpus_fpath)
    
    if detect_bigrams:
        print("Extracting bigrams from the corpus:", corpus_fpath)

        bigram_transformer = Phrases(sentences, min_count=min_count)
        bigrams = Phraser(bigram_transformer)
        sentences = list(bigrams[sentences])
        print("Time, sec.:", time() - tic)

    if exists(phrases_fpath):
        tic = time()
        print("Finding phrases from the input dictionary:", phrases_fpath)

        bigram_transformer = load_vocabulary(phrases_fpath)
        sentences_tmp = sentences
        sentences = [add_phrases(sentence, bigram_transformer, detect_bigrams) for sentence in sentences_tmp]

        print("Time, sec.:", time() - tic)


    print("Training word vectors:", corpus_fpath)
    model = Word2Vec(sentences,
                     min_count=min_count,
                     size=size,
                     window=window, 
                     max_vocab_size=None,
                     workers=threads,
                     sg=(1 if cbow == 0 else 0),
                     iter=iter_num)
    model.wv.save_word2vec_format(vectors_fpath, binary=False)
    print("Vectors:", vectors_fpath)
    print("Time, sec.:", time()-tic) 

