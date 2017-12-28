# coding=utf-8
import argparse, codecs
from sys import stderr, stdin, stdout
import re
from time import time
import numpy as np
from collections import OrderedDict, defaultdict
import sys, traceback
from parallel import parallel_map
import gensim
from math import ceil
from sys import stderr


re_only_letters = re.compile(u'^[a-zA-Z\.\-]+$')


def load_freq(freq_file):
    print "Loading frequencies"
    d = defaultdict(int)
    with codecs.open(freq_file, 'r', encoding='utf-8') as f:
        for line in f:
            (key, val) = line.split()
            d[key] = int(val)
    return d

def similar_top(vec, words, topn=200):
    res = OrderedDict()
    for word in words:
        res[word] = vec.most_similar(positive=[word],negative=[], topn=topn)
    return res

def argmax_k(dists, topn):
    dists = -dists
    return np.argpartition(dists, topn,axis=1)[:,:topn]

def similar_top_opt(vec, words, topn=200):
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
    # dist shape is (current_batch x vocabulary_size)
    best = argmax_k(dists,topn)

    res = OrderedDict()
    for i in xrange(len(indices)):
        sims = best[i,np.argsort(-dists[i, best[i]])] # sims is a list of indices (in relation to syn0norm) of nearest neighbours
                                                      # sorted(!) by similarity
        ns = [(vec.index2word[sim], float(dists[i, sim])) for sim in sims if sim!=indices[i]]
        res[vec.index2word[indices[i]]] = ns
    return res

def order_freq(vec, freq):
    "return frequencies of words as an array ordered excatly as words in vec.syn0norm"
    l = []
    for i in xrange(len(vec.syn0norm)):
        if freq[vec.index2word[i]] > 0:
            l.append(freq[vec.index2word[i]])
        else: 
            l.append(1) # neutral frequency for words with unknown frequency

    return np.array(l)
    
def similar_top_opt3(vec, words, topn=200, nthreads=12, freq=None):
    vec.init_sims()

    indices = [vec.vocab[w].index for w in words if w in vec.vocab]
    vecs = vec.syn0norm[indices]
    dists = np.dot(vecs, vec.syn0norm.T)
    
    if freq is not None:
        dists = dists * np.log(freq)

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


def print_similar(out, vectors, batch, mindist=None, only_letters=False, threads_num=4, pairs=False, freq=None):
    try:
        for word, ns in similar_top_opt3(vectors, batch, nthreads=threads_num, freq=freq).iteritems():
            sims = []
            for w, d in ns:
                if (mindist is None or d >= mindist) and (not only_letters or re_only_letters.match(w) is not None):
                    sims.append((w, d))
                else:
                    print >> stderr,  "%s: SKIPPED\t%s\t%r" % (word.encode('utf8'), w.encode('utf8'), d)

            if pairs:
                print >> out, '\n'.join(("%s\t%s\t%f" % (word.encode('utf8'), w.encode('utf8'), d) for w, d in sims))
            else:
                print >> out, "%s\t%s" % (word.encode('utf8'), ','.join(("%s:%f" % (w.encode('utf8'), d) for w, d in sims)))
    except:
        print >> stderr, "ERROR in print_similar()"
        traceback.print_exc(file=sys.stderr)


def process(output_file, vectors, words, only_letters, batch_size=10000, threads_num=4, pairs=False, freq=None):
    batch = []
    for word in words:
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
            print_similar(output_file, vectors, batch, only_letters=only_letters, threads_num=threads_num, pairs=pairs, freq=freq)
            batch = []

    if len(batch) > 0:
        print_similar(output_file, vectors, batch, only_letters=only_letters, pairs=pairs, freq=freq)

def run(vectors_fpath, output_fpath="", only_letters=False, vocab_limit=None, pairs=False, batch_size=1000, threads_num=4, word_freqs=None):
    print >> stderr, "Vectors: {}, only_letters: {}".format(vectors_fpath, only_letters)
    print >> stderr, "Loading vectors from {}".format(vectors_fpath)
    tic = time()
    vectors = gensim.models.KeyedVectors.load_word2vec_format(
        vectors_fpath, binary=False, unicode_errors='ignore')
    vectors.init_sims(replace=True)

    print >> stderr, "Vectors loaded in %d sec." % (time()-tic)
    print >> stderr, "Vectors shape is: ", vectors.syn0norm.shape

    vocab_size = len(vectors.vocab)
    print("Vocabulary size: %i" % vocab_size)
    
    # Limit the number of words for which to collect neighbours
    if vocab_limit and vocab_limit < vocab_size:
        vocab_size = vocab_limit
    words = vectors.index2word[:vocab_size]
    
    print("Collect neighbours for %i most frequent words" % vocab_size)
    
    freq=None
    if word_freqs:
        freq_dict = load_freq(word_freqs)
        freq = order_freq(vectors, freq_dict)
        print "freqs loaded. Length ", len(freq), freq[:10]

    with codecs.open(output_fpath, 'wb') if output_fpath else stdout as output_file:
        process(output_file, vectors, words, only_letters=only_letters, batch_size=batch_size, threads_num=threads_num, pairs=pairs, freq=freq)

def main():
    parser = argparse.ArgumentParser(
        description='Efficient computation of nearest word neighbours. Reads words from a vector model. '
                    'Writes to output word and its similar words and their distances to the original word.')
    parser.add_argument('vectors', help='Word2vec word vectors file.', default='')
    parser.add_argument('-output', help='Output file in on-pair-per-line format, gziped', default='')
    parser.add_argument('-only_letters', help='Skip words containing non-letter symbols from stding / similar words.', action="store_true")
    parser.add_argument("-vocab_limit", help="Collect neighbours only for specified number of most frequent words. By default use all words.", default=None, type=int)
    parser.add_argument('-pairs', help="Use pairs format: 2 words and distance in each line. Otherwise echo line is a word and all it's neighbours with distances." , action="store_true")
    parser.add_argument('-batch-size', help='Batch size for finding neighbours.', default="1000")
    parser.add_argument('-word_freqs', help="Weight similar words by frequency. Pass frequency file as parameter", default=None)
    args = parser.parse_args()
     
    run(args.vectors, output_fpath=args.output, only_letters=args.only_letters, vocab_limit=args.vocab_limit, pairs=args.pairs, batch_size=int(args.batch_size), word_freqs=args.word_freqs)

if __name__ == '__main__':
    main()
