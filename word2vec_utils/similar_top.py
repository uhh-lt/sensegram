# coding=utf-8
import argparse
import gzip
from sys import stderr, stdin, stdout
from utils import load_vectors
import re
from time import time
import numpy as np
from collections import OrderedDict
import sys, traceback
from parallel import parallel_map

from math import ceil
from sys import stderr
__author__ = 'nvanva'

re_only_letters = re.compile(u'^[а-яА-ЯёЁ]+$')

def similar_top(vec, words, topn=250):
    res = OrderedDict()
    for word in words:
        res[word] = vec.most_similar(positive=[word],negative=[], topn=topn)
    return res

def argmax_k(dists, topn):
    dists = -dists
    return np.argpartition(dists, topn,axis=1)[:,:topn]

def similar_top_opt(vec, words, topn=250):
    vec.init_sims()

    indices = [vec.vocab[w].index for w in words if w in vec.vocab]
    vecs = vec.syn0norm[indices]
    dists = np.dot(vecs, vec.syn0norm.T)
    best = argmax_k(dists,topn)

    res = OrderedDict()
    for i in xrange(len(indices)):
        sims = best[i,np.argsort(-dists[i, best[i]])]
        ns = [(vec.index2word[sim], float(dists[i, sim])) for sim in sims if sim!=indices[i]]
        res[vec.index2word[indices[i]]] = ns

    return res


def dists2neighbours(vec, dists, indices, topn):
    best = argmax_k(dists,topn)

    res = OrderedDict()
    for i in xrange(len(indices)):
        sims = best[i,np.argsort(-dists[i, best[i]])]
        ns = [(vec.index2word[sim], float(dists[i, sim])) for sim in sims if sim!=indices[i]]
        res[vec.index2word[indices[i]]] = ns
    return res


def similar_top_opt3(vec, words, topn=250, nthreads=4):
    vec.init_sims()

    indices = [vec.vocab[w].index for w in words if w in vec.vocab]
    vecs = vec.syn0norm[indices]
    dists = np.dot(vecs, vec.syn0norm.T)

    if nthreads==1:
        res = dists2neighbours(vec, dists, indices, topn)
    else:
        batchsize = int(ceil(1. * len(indices) / nthreads))
        print >> stderr, "dists2neighbours for %d words in %d threads, batchsize=%d" % (len(indices), nthreads, batchsize)
        def ppp(i):
            return dists2neighbours(vec, dists[i:i+batchsize], indices[i:i+batchsize], topn)
        lres = parallel_map(ppp, range(0,len(indices),batchsize), threads=nthreads)
        res = OrderedDict()
        for lr in lres:
            res.update(lr)

    return res


def print_similar(out, vectors, batch, mindist=None, only_letters=False, pairs=False):
    try:
        for word, ns in similar_top_opt3(vectors, batch).iteritems():
            sims = []
            for w, d in ns:
                if (mindist is None or d >= mindist) and (not only_letters or re_only_letters.match(w) is not None):
                    # print >> stderr, "%s: RETURNED\t%s\t%r" % (word.encode('utf8'), w.encode('utf8'), sim)
                    sims.append((w, d))
                else:
                    print >> stderr,  "%s: SKIPPED\t%s\t%r" % (word.encode('utf8'), w.encode('utf8'), d)

            if pairs:
                print >> out, '\n'.join(("%s\t%s\t%f" % (word.encode('utf8'), w.encode('utf8'), d) for w, d in sims))
            else:
                print >> out, "%s\t%s" % (word.encode('utf8'), ','.join(("%s:%f" % (w.encode('utf8'), d) for w, d in sims)))

            #print >> stderr, "%s: %d similar words found" % (word.encode('utf8'), len(sims))
    except:
        print >> stderr, "ERROR in print_similar()"
        traceback.print_exc(file=sys.stderr)


def process(out, vectors, only_letters, vocab_size, batch_size=1000, pairs=False):
    batch = []
    for word in vectors.index2word[:vocab_size]:
        try:
            word = word.decode('utf8').rstrip('\n')
        except UnicodeDecodeError:
            print >> stderr, "couldn't decode word from stdout, skipped"
            continue
        if only_letters and re_only_letters.match(word) is None:
            print >> stderr, "%s: SKIPPED_ALL" % word.encode('utf8')
            continue

        batch.append(word)

        if len(batch) >=  batch_size:
            print_similar(out, vectors, batch, pairs=pairs)
            batch = []

    if len(batch) > 0:
        print_similar(out, vectors, batch, pairs=pairs)


def main():
    parser = argparse.ArgumentParser(
        description='Reads words from vector model. Writes to stdout word + similar words and their distances to the original word.')
    parser.add_argument('vectors', help='word2vec word vectors file.', default='')
    parser.add_argument('-output', help='Output file in on-pair-per-line format, gziped', default='')
    parser.add_argument('-only_letters', help='Skip words containing non-letter symbols from stding / similar words.', action="store_true")
    parser.add_argument("-vocab_limit", help="Collect neighbours only for specified number of most frequent words. By default use all words.", default=None, type=int)
    parser.add_argument('-pairs', help="Use pairs format: 2 words and distance in each line. Otherwise echo line is a word and all it's neighbours with distances." , action="store_true")
    parser.add_argument('-batch-size', help='Batch size for finding neighbours.', default="1000")

    args = parser.parse_args()

    fvec = args.vectors
    batch_size = int(args.batch_size)

    print >> stderr, "Vectors: {}, only_letters: {}".format(args.vectors, args.only_letters)
    print >> stderr, "Loading vectors from {}".format(fvec)
    tic = time()
    vectors = load_vectors(fvec)
    print >> stderr, "Vectors loaded in %d sec." % (time()-tic)
    print >> stderr, "Vectors shape is: ", vectors.syn0norm.shape

    vocab_size = len(vectors.vocab)
    print("Vocabulary size: %i" % vocab_size)
    
    # Limit the number of words for which to collect neighbours
    if args.vocab_limit and args.vocab_limit < vocab_size:
        vocab_size = vocab_limit
        
    print("Collect neighbours for %i most frequent words" % vocab_size)

    with gzip.open(args.output, 'wb') if args.output else stdout as out:
        process(out, vectors, only_letters=args.only_letters, vocab_size=vocab_size, batch_size=batch_size, pairs=args.pairs)


if __name__ == '__main__':
    main()
