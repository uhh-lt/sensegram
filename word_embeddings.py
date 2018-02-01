import gzip
import codecs
from gensim.models.keyedvectors import KeyedVectors
from gensim.utils import tokenize
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec


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


def learn_word_embeddings(corpus_fpath, vectors_fpath, cbow, window, iter_num, size, threads, min_count, detect_phrases=True):

    tic = time()
    sentences = GzippedCorpusStreamer(corpus_fpath) 
    
    if detect_phrases:
        print("Extracting phrases from the corpus:", corpus_fpath)
        phrases = Phrases(sentences)
        bigram = Phraser(phrases)
        input_sentences = list(bigram[sentences])
        print("Time, sec.:", time()-tic)
    else:
        input_sentences = sentences
    
    print("Training word vectors:", corpus_fpath)
    print(threads) 
    model = Word2Vec(input_sentences,
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

