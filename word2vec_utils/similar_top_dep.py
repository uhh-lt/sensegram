# coding=utf-8
import argparse, codecs
from sys import stderr, stdin, stdout
from utils import load_vectors
import re
from time import time
import numpy as np
from collections import OrderedDict, defaultdict
import sys, traceback
from parallel import parallel_map

from math import ceil
from sys import stderr
__author__ = 'nvanva'

re_only_letters = re.compile(u'^[a-zA-Z\.\-]+$')

def load_freq(freq_file):
    print "Loading frequencies"
    d = defaultdict(int)
    with codecs.open(freq_file, 'r', encoding='utf-8') as f:
        for line in f:
            (key, val) = line.split()
            d[key] = int(val)
    return d

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


def dists2neighbours(wvectors, cvectors, dists, indices, topn):
    # dist shape is (current_batch x vocabulary_size)
    best = argmax_k(dists,topn)

    res = OrderedDict()
    for i in xrange(len(indices)):
        sims = best[i,np.argsort(-dists[i, best[i]])] # sims is a list of indices (in relation to syn0norm) of nearest neighbours
                                                      # sorted(!) by similarity
        ns = [(cvectors.index2word[sim], float(dists[i, sim])) for sim in sims if sim!=indices[i]]
        res[wvectors.index2word[indices[i]]] = ns
    return res

    
def similar_top_opt3(wvectors, cvectors, words, topn=200, nthreads=4):
    wvectors.init_sims()
    cvectors.init_sims()
    
    indices = [wvectors.vocab[w].index for w in words if w in wvectors.vocab]
    wvecs = wvectors.syn0norm[indices]
    dists = np.dot(wvecs, cvectors.syn0norm.T)
    

    if nthreads==1:
        res = dists2neighbours(wvectors, cvectors, dists, indices, topn)
    else:
        batchsize = int(ceil(1. * len(indices) / nthreads))
        print >> stderr, "dists2neighbours for %d words in %d threads, batchsize=%d" % (len(indices), nthreads, batchsize)
        def ppp(i):
            return dists2neighbours(wvectors, cvectors, dists[i:i+batchsize], indices[i:i+batchsize], topn)
        lres = parallel_map(ppp, range(0,len(indices),batchsize), threads=nthreads)
        res = OrderedDict()
        for lr in lres:
            res.update(lr)

    return res


def print_similar(out, wvectors, cvectors, batch, mindist=None, only_letters=False, pairs=False):
    try:
        for word, ns in similar_top_opt3(wvectors, cvectors, batch).iteritems():
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
                print >> out, "%s\t%s" % (word.encode('utf8'), ' '.join(("%s@%f" % (w.encode('utf8'), d) for w, d in sims)))

            #print >> stderr, "%s: %d similar words found" % (word.encode('utf8'), len(sims))
    except:
        print >> stderr, "ERROR in print_similar()"
        traceback.print_exc(file=sys.stderr)


def process(out, wvectors, cvectors, words, only_letters, batch_size=1000, pairs=False):
    batch = []
    for word in words:#vectors.index2word[:vocab_size]:
        try:
            word = word.rstrip('\n')
        except UnicodeDecodeError:
            print >> stderr, "couldn't decode word from stdout, skipped"
            continue
        if only_letters and re_only_letters.match(word) is None:
            print >> stderr, "%s: SKIPPED_ALL" % word
            continue

        batch.append(word)

        if len(batch) >=  batch_size:
            print_similar(out, wvectors, cvectors, batch, only_letters=only_letters, pairs=pairs)
            batch = []

    if len(batch) > 0:
        print_similar(out, wvectors, cvectors, batch, only_letters=only_letters, pairs=pairs)

def init(wvec, cvec, output="", only_letters=False, vocab_limit=None, pairs=False, batch_size=1000):

    print >> stderr, "Vectors: {}, only_letters: {}".format(wvec, only_letters)
    print >> stderr, "Loading vectors from {}".format(wvec)
    tic = time()
    wvectors = load_vectors(wvec, binary=False)
    print >> stderr, "Vectors loaded in %d sec." % (time()-tic)
    print >> stderr, "Vectors shape is: ", wvectors.syn0norm.shape
    
    print >> stderr, "Loading vectors from {}".format(cvec)
    tic = time()
    cvectors = load_vectors(cvec, binary=False)
    print >> stderr, "Vectors loaded in %d sec." % (time()-tic)
    print >> stderr, "Vectors shape is: ", cvectors.syn0norm.shape
    

    vocab_size = len(wvectors.vocab)
    print("Vocabulary size: %i" % vocab_size)
    
    # Limit the number of words for which to collect contexts
    if vocab_limit and vocab_limit < vocab_size:
        vocab_size = vocab_limit
    words = wvectors.index2word[:vocab_size]
    
    print("Collect activated contexts for %i most frequent words" % vocab_size)
    

    with codecs.open(output, 'wb') if output else stdout as out:
        process(out, wvectors, cvectors, words, only_letters=only_letters, batch_size=batch_size, pairs=pairs)

def main():
    parser = argparse.ArgumentParser(
        description='Efficient computation of activated contexts for words.')
    parser.add_argument('wvectors', help='Word2vec word vectors file.', default='')
    parser.add_argument('cvectors', help='Word2vec context vectors file.', default='')
    parser.add_argument('-output', help='Output file. Format: "word<TAB>cluster" where <cluster> is "context@sim context@sim,..."', default='')
    parser.add_argument('-only_letters', help='Skip words containing non-letter symbols from stding / similar words.', action="store_true")
    parser.add_argument("-vocab_limit", help="Collect neighbours only for specified number of most frequent words. By default use all words.", default=None, type=int)
    parser.add_argument('-pairs', help="Use pairs format: 2 words and distance in each line. Otherwise echo line is a word and all it's neighbours with distances. (not in use here)" , action="store_true")
    parser.add_argument('-batch-size', help='Batch size for finding activated contexts.', default="1000")

    args = parser.parse_args()
     
    init(args.wvectors, args.cvectors, output=args.output, only_letters=args.only_letters, vocab_limit=args.vocab_limit, pairs=args.pairs, batch_size=int(args.batch_size))

if __name__ == '__main__':
    main()
