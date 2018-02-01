import faiss
import codecs
from time import time
from gensim.models import KeyedVectors


def compute_graph_of_related_words(vectors_fpath, neighbours_fpath, neighbors=200):
    print("Start collection of word neighbours.")
    tic = time()
    index, w2v = build_vector_index(vectors_fpath)
    compute_neighbours(index, w2v, neighbours_fpath, neighbors)
    print("Elapsed: {:f} sec.".format(time() - tic))


def build_vector_index(w2v_fpath):
    w2v = KeyedVectors.load_word2vec_format(w2v_fpath, binary=False, unicode_errors='ignore')
    w2v.init_sims(replace=True)
    index = faiss.IndexFlatIP(w2v.vector_size)
    index.add(w2v.syn0norm)

    return index, w2v


def compute_neighbours(index, w2v, nns_fpath, neighbors=200):
    tic = time()
    with codecs.open(nns_fpath, "w", "utf-8") as output:
        X = w2v.syn0norm
        D, I = index.search(X, neighbors + 1)

        j = 0
        for _D, _I in zip(D, I):
            for n, (d, i) in enumerate(zip(_D.ravel(), _I.ravel())):
                if n > 0:
                    output.write("{}\t{}\t{:f}\n".format(w2v.index2word[j], w2v.index2word[i], d))
            j += 1

        print("Word graph:", nns_fpath)
        print("Elapsed: {:f} sec.".format(time() - tic))


