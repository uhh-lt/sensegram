#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reads sense clusters (output of chinese-whispers) and creates a sense vector for each cluster 
(currently by averaging vectors of words in the cluster).
Result - a model of sense vectors (python#0, python#1, etc) saved in word2vec binary format.
If -inventory path is set, also creates an inventory for this sense vector model.
"""

# Currently creates vectors for clusters in input.
# What if: 1. A word from original model doesn't have a cluster left after postprocessing 
# (all cluster were too small)
# 2. A word has only one cluster. Use the average of cluster (as currently) or the original word vector?
# TODO: duplicates check not necessary?
"""
1. Throw words out of cluster if they cannot be parsed correctly 
2. Cluster center word isn't lowercased
3. Cluster words are lowercased only if the word vector model (which provides vectors for averaging) is universally lowercase
4. 
"""

# wc -l <filename> number of lines in a file

import argparse, codecs
from operator import methodcaller
from collections import defaultdict
import numpy as np
from pandas import read_csv
from gensim.models import word2vec
import pbar

CHUNK_LINES = 500000
debug = True
default_count = 100 # arbitrary, should be larger than min_count of vec object, which is 5 by default
sen_delimiter = u"#" # python#0, python#1, etc
inventory_header = u"word\tsense_id\trel_terms\n"

def pool_vectors(vectors, similarities, method):
    if method == 'mean':
        return np.mean(vectors, axis=0)
    if method == 'weighted_mean':
        s = sum(similarities)
        sim_weights = [sim/s for sim in similarities]
        return np.average(vectors, axis=0, weights=sim_weights)
    else:
        raise ValueError("Unknown pooling method '%s'" % method)

class Dummysink(object):
    # dummy object imitates file object but does nothing.
    def write(self, data):
        pass # ignore the data
    def __enter__(self):
        return self
    def __exit__(self, *x):
        return False

def write_inventory(filename):
    # If inventory filename is set, open it and write the inventory. Otherwise do nothing.
    if filename:
        return codecs.open(filename, "w", encoding='utf-8')
    else:
        return Dummysink()

def run(clusters, nclusters, model, output, method='mean', format='word2vec', binary=True, lowercase=False, inventory=None):

    duplicates = 0 
    small_clusters = 0 

    print("Loading original word model...")
    if format == 'word2vec':
        wordvec = word2vec.Word2Vec.load_word2vec_format(model, binary=binary)
    if format == 'gensim':
        wordvec = word2vec.Word2Vec.load(model)
    vector_size = wordvec.syn0.shape[1]
    if debug: print("Vector size = %i" % vector_size)

    print("Initializing Word2Vec object for sense vectors...")
    senvec = word2vec.Word2Vec(size=vector_size, sorted_vocab=0)
    senvec.syn0 = np.zeros((nclusters, vector_size), dtype=np.float32)

    print("Pooling cluster vectors...")
    # na_values=[""], keep_default_na=False means that strings 'NaN', 'nan', 'na' etc will be interpreted 
    # as corresponding strings, not replaced with float NaN.
    # doublequote=False, quotechar=u"\u0000" changes quotechar from default '"' to NUL
    # otherwise any delimiter inside quotes would be ignores
    reader = read_csv(clusters, encoding="utf-8", delimiter="\t", error_bad_lines=False, iterator=True, chunksize=CHUNK_LINES, 
                      na_values=[""], keep_default_na=False, doublequote=False, quotechar=u"\u0000")
    sen_count = defaultdict(int)
    pb = pbar.Pbar(nclusters, 100)
    pb.start()

    with write_inventory(inventory) as inv_output:
        inv_output.write(inventory_header)
        i = 0
        for chunk in reader:
            if debug: print("Column types: %s" % chunk.dtypes)
            for j, row in chunk.iterrows():
                row_word = row.word
                row_cluster = row.cluster
                if lowercase:
                    # row_word = row_word.lower()
                    row_cluster = row_cluster.lower()
                # enumerate word senses from 0
                sen_word = unicode(row_word) + sen_delimiter + unicode(sen_count[row_word])
                
                # only pool cluster words which are in the word vector model
                # skip words in clusters that cannot be split correctly
                cluster_words = []
                for cluster_word_entry in row_cluster.split(','):
                    try:
                        word, sim = cluster_word_entry.strip().rsplit(':', 1)
                        if word in wordvec.vocab:
                            cluster_words.append((word, float(sim)))
                    except:
                        print "Warning: wrong cluster word", cluster_word_entry
                
                # copied from word2vec, modified
                def add_word(word, vector):
                    "add new word to the model"
                    word_id = len(senvec.vocab)
                    if word in senvec.vocab:
                        print("duplicate word sense '%s' in %s, ignoring all but first" % (word, clusters_path))
                        duplicate += 1
                        return
                    senvec.vocab[word] = word2vec.Vocab(index=word_id, count=default_count)
                    senvec.syn0[word_id] = vector
                    senvec.index2word.append(word)
                    assert word == senvec.index2word[senvec.vocab[word].index]

                if len(cluster_words) >= 5:
                    cluster_vectors = np.array([wordvec[word] for word, sim in cluster_words])
                    cluster_sim = np.array(sim for word, sim in cluster_words])
                    sen_vector = pool_vectors(cluster_vectors, cluster_sim, method)
                    add_word(sen_word, sen_vector)
                    if inventory:
                        # join back cluster words (only those that were actually used for sense vector)
                        cluster = ",".join([word + ":" + sim for word, sim in cluster_words])
                        inv_output.write(u"%s\t%s\t%s\n" % (sen_word.split(sen_delimiter)[0], 
                                                            sen_word.split(sen_delimiter)[1], cluster))
                    sen_count[row_word] += 1
                else: 
                    small_clusters += 1
                    if debug:
                        print row_word, "\t", row.cid
                        print cluster_words
                pb.update(i)
                i += 1 
        pb.finish()

    ##### Validation #####
    if senvec.syn0.shape[0] != len(senvec.vocab):
        print("Shrinking matrix size from %i to %i" % (senvec.syn0.shape[0], len(senvec.vocab)))
        senvec.syn0 = np.ascontiguousarray(senvec.syn0[:len(senvec.vocab)])
    assert (len(senvec.vocab), senvec.vector_size) == senvec.syn0.shape
    print("Sense vectors: %i, duplicates: %i, small: %i, clusters: %i" % 
                    (len(senvec.vocab), duplicates, small_clusters, nclusters))
    assert (len(senvec.vocab) + duplicates + small_clusters == nclusters)

    print("Saving sense vectors...")
    senvec.save_word2vec_format(fname=output, binary=True)



def main():
    parser = argparse.ArgumentParser(description='Create sense vectors based on sense clusters and word vectors.')
    parser.add_argument('clusters', help='A path to an input file with postprocessed clusters and a header. Format: "word<TAB>cid<TAB>cluster" where <cluster> is "word:sim,word:sim,..."')
    # TODO: make default=None and implement counting clusters in the input file
    parser.add_argument('nclusters', help='number of clusters in the input file.', type=int)
    parser.add_argument('model', help='A path to an initial word vector model')
    parser.add_argument('output', help='A path to the output sense vector model')
    parser.add_argument('-method', help="A method used to pool word vectors into a sense vector ('mean' or 'weighted_mean'). Default 'mean'", default='mean')
    parser.add_argument("-format", help="model type:'word2vec' or 'gensim'. Default 'word2vec'.", default="word2vec")
    parser.add_argument("-binary", help="1 for binary model, 0 for text model. Applies to word2vec only. Default 1", default=1, type=int)
    parser.add_argument("-lowercase", help="Lowercase all words in clusters (necessary if word model only has lowercased words). Default False", action="store_true")
    parser.add_argument("-inventory", help='A path to the output inventory file of computed sense vector model with a header. Format: "word<TAB>sense_id<TAB>rel_terms" where <rel_terms> is "word:sim,word:sim,...". If not given, inventory is not written. Default None', default=None)
    args = parser.parse_args()

    run(args.clusters, int(args.nclusters), args.model, args.output, args.method, args.format, args.binary, args.lowercase, args.inventory) 
    
if __name__ == '__main__':
    main()
    