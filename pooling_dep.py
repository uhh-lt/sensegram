#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, codecs
from operator import methodcaller
from collections import defaultdict
import numpy as np
from pandas import read_csv
from gensim.models import word2vec
import sensegram
import pbar

CHUNK_LINES = 500000
SPLIT_MWE = True
debug = False

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def pool_vectors(vectors, similarities, method):
    if method == 'mean':
        return np.mean(vectors, axis=0)
    if method == 'weighted_mean':
        s = sum(similarities)
        sim_weights = [sim/s for sim in similarities]
        return np.average(vectors, axis=0, weights=sim_weights)
    if method == 'ranked':
        cluster_size = len(vectors)
        rank_weights = [1/rank for rank in range(1, cluster_size + 1)]
        return np.average(vectors, axis=0, weights=rank_weights)
    else:
        raise ValueError("Unknown pooling method '%s'" % method)
    
def initialize(clusters_file, has_header, vector_size):
    """ Initialize sense model """
    nclusters = file_len(clusters_file)
    if has_header:
        ncluster = nclusters - 1
    senvec = sensegram.SenseGram(size=vector_size, sorted_vocab=0)
    senvec.syn0 = np.zeros((nclusters, vector_size), dtype=np.float32)
    if debug: 
        print("Matrix shape: (%i, %i)" % (nclusters, vector_size))
    return senvec

def read_clusetrs_file(clusters, has_header):
    # na_values=[""], keep_default_na=False means that strings 'NaN', 'nan', 'na' etc will be interpreted 
    # as corresponding strings, not replaced with float NaN.
    # doublequote=False, quotechar=u"\u0000" changes quotechar from default '"' to NUL
    # otherwise any delimiter inside quotes would be ignores
    if has_header:
        reader = read_csv(clusters, encoding="utf-8", delimiter="\t", error_bad_lines=False, iterator=True,
                          chunksize=CHUNK_LINES, na_values=[""], keep_default_na=False, 
                          doublequote=False, quotechar=u"\u0000", index_col=False)
    else:
        reader = read_csv(clusters, encoding="utf-8", delimiter="\t", error_bad_lines=False, iterator=True,
                          chunksize=CHUNK_LINES, na_values=[""], keep_default_na=False, 
                          doublequote=False, quotechar=u"\u0000",
                          header=None, names=["word","cluster"])
    return reader

def parse_cluster(row_cluster, contextvec):
    # only pool cluster words which are in the word vector model
    # skip words in clusters that cannot be split correctly
    cluster = []
    for item in row_cluster.split(' '):
        try:
            word, sim = item.strip().rsplit('@', 1)
            float(sim) # assert sim string represents a float
            if word in contextvec.vocab:
                cluster.append((word, sim))
        except:
            print "Warning: wrong cluster word", item
    return cluster
    
def run(clusters, model, n, output, method='weighted', has_header=True):

    print("Loading original context model...")
    contextvec = word2vec.Word2Vec.load_word2vec_format(model, binary=False)
    print("Initializing new word model...")
    wordvec = initialize(clusters, has_header, contextvec.syn0.shape[1])

    print("Pooling cluster vectors (%s method)..." % method)
    reader = read_clusetrs_file(clusters, has_header)
    
    
    pb = pbar.Pbar(wordvec.syn0.shape[0], 100)
    pb.start()
    i = 0
    for chunk in reader:
        if debug: 
            print("Column types: %s" % chunk.dtypes)
        for j, row in chunk.iterrows():
            row_word = row.word
            row_cluster = row.cluster

            # process new word
            word_cluster = parse_cluster(row_cluster, contextvec)[:n]

            vectors = np.array([contextvec[context] for context, sim in word_cluster])
            sims = np.array([float(sim) for context, sim in word_cluster])
            word_vector = pool_vectors(vectors, sims, method)

            if row_word not in wordvec.vocab:
                wordvec.add_word(row_word, word_vector)
                
            pb.update(i)
            i += 1 
    pb.finish()

    ##### Validation #####
    if wordvec.syn0.shape[0] != len(wordvec.vocab):
        print("Shrinking matrix size from %i to %i" % (wordvec.syn0.shape[0], len(wordvec.vocab)))
        wordvec.syn0 = np.ascontiguousarray(wordvec.syn0[:len(wordvec.vocab)])
    print("Sense vectors saved to: " + output)
    wordvec.save_word2vec_format(fname=output, binary=True)

def main():
    parser = argparse.ArgumentParser(description='Create word vectors based on n context vectors.')
    parser.add_argument('clusters', help='A path to an input file with clusters and a header. Format: "word<TAB>cluster" where <cluster> is "context@sim context@sim,..."')
    parser.add_argument('model', help='A path to an initial context vector model')
    parser.add_argument('n', help='Number of contexts to use for calculation of a word vector', type = int)
    parser.add_argument('output', help='A path to the output word vector model')
    parser.add_argument('-method', help="A method used to pool context vectors into a word vector ('mean', 'weighted_mean', 'ranked'). Default 'weighted_mean'", default='weighted_mean')
    parser.add_argument('-no_header', action='store_true', help='No headers in cluster file. Default -- false.')
    args = parser.parse_args()

    run(args.clusters, args.model, args.n, args.output, args.method, has_header=(not args.no_header)) 
    
if __name__ == '__main__':
    main()
    