#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reads sense inventory (chinese-whispers format: word<TAB>sense_id<TAB>cluster, where cluster= word:weight,word:weight) 
and creates a sense vector for each cluster.
Result - a sensegram model, each sense in form word#sense_id. 
If -inventory path is set, also creates a new sense inventory for this sense vector model.
"""

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
sen_delimiter = u"#" # python#0, python#1, etc
inventory_header = u"word\tsense_id\trel_terms\n"

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
                          header=None, names=["word","cid","cluster"])
    return reader

def parse_cluster(row_cluster, wordvec):
    # only pool cluster words which are in the word vector model
    # skip words in clusters that cannot be split correctly
    cluster = []
    for item in row_cluster.split(','):
        try:
            word, sim = item.strip().rsplit(':', 1)
            float(sim) # assert sim string represents a float
            if word in wordvec.vocab:
                cluster.append((word, sim))
            if SPLIT_MWE:
                words = word.split(" ")
                if len(words) == 1: 
                    continue
                for w in words:
                    if w in wordvec.vocab: 
                        cluster.append((w, sim))
        except:
            print "Warning: wrong cluster word", item
    return cluster
    
def run(clusters, model, output, method='weighted', lowercase=False, inventory=None, has_header=True):

    small_clusters = 0 
    sen_count = defaultdict(int)   # number of senses per word
    cluster_sum = defaultdict(int) # number of cluster words per word

    print("Loading original word model...")
    wordvec = word2vec.Word2Vec.load_word2vec_format(model, binary=True)
    print("Initializing sense model...")
    senvec = initialize(clusters, has_header, wordvec.syn0.shape[1])

    print("Pooling cluster vectors (%s method)..." % method)
    reader = read_clusetrs_file(clusters, has_header)
    
    
    pb = pbar.Pbar(senvec.syn0.shape[0], 100)
    pb.start()

    with write_inventory(inventory) as inv_output:
        inv_output.write(inventory_header)
        i = 0
        for chunk in reader:
            if debug: 
                print("Column types: %s" % chunk.dtypes)
            for j, row in chunk.iterrows():
                row_word = row.word
                row_cluster = row.cluster
                
                if lowercase:
                    row_cluster = row_cluster.lower()
                    
                # enumerate word senses from 0
                sen_word = unicode(row_word) + sen_delimiter + unicode(sen_count[row_word])
                
                # process new sense
                sen_cluster = parse_cluster(row_cluster, wordvec)
                if len(sen_cluster) >= 5:
                    vectors = np.array([wordvec[word] for word, sim in sen_cluster])
                    sims = np.array([float(sim) for word, sim in sen_cluster])
                    sen_vector = pool_vectors(vectors, sims, method)
                    
                    if sen_word not in senvec.vocab:
                        senvec.add_word(sen_word, sen_vector)
                        senvec.probs[sen_word] = len(sen_cluster) # number of cluster words per sense
                        sen_count[row_word] += 1                  # number of senses per word
                        cluster_sum[row_word] += len(sen_cluster) # number of cluster words per word
                    
                    # write new sense to sense inventory
                    if inventory:
                        # join back cluster words (only those that were actually used for sense vector)
                        cluster = ",".join([word + ":" + sim for word, sim in sen_cluster])
                        inv_output.write(u"%s\t%s\t%s\n" % (sen_word.split(sen_delimiter)[0], 
                                                            sen_word.split(sen_delimiter)[1], cluster))
                else: 
                    small_clusters += 1
                    if debug:
                        print row_word, "\t", row.cid
                        print sen_cluster
                pb.update(i)
                i += 1 
        senvec.__normalize_probs__(cluster_sum)
        pb.finish()

    ##### Validation #####
    if senvec.syn0.shape[0] != len(senvec.vocab):
        print("Shrinking matrix size from %i to %i" % (senvec.syn0.shape[0], len(senvec.vocab)))
        senvec.syn0 = np.ascontiguousarray(senvec.syn0[:len(senvec.vocab)])
    print("Sense vectors saved to: " + output)
    senvec.save_word2vec_format(fname=output, binary=True)

    



def main():
    parser = argparse.ArgumentParser(description='Create sense vectors based on sense clusters and word vectors.')
    parser.add_argument('clusters', help='A path to an input file with postprocessed clusters and a header. Format: "word<TAB>cid<TAB>cluster" where <cluster> is "word:sim,word:sim,..."')
    parser.add_argument('model', help='A path to an initial word vector model')
    parser.add_argument('output', help='A path to the output sense vector model')
    parser.add_argument('-method', help="A method used to pool word vectors into a sense vector ('mean', 'weighted_mean', 'ranked'). Default 'weighted_mean'", default='weighted_mean')
    parser.add_argument("-lowercase", help="Lowercase all words in clusters (necessary if word model only has lowercased words). Default False", action="store_true")
    parser.add_argument("-inventory", help='A path to the output inventory file of computed sense vector model with a header. Format: "word<TAB>sense_id<TAB>rel_terms" where <rel_terms> is "word:sim,word:sim,...". If not given, inventory is not written. Default None', default=None)
    parser.add_argument('-no_header', action='store_true', help='No headers in cluster file. Default -- false.')
    args = parser.parse_args()

    run(args.clusters, args.model, args.output, args.method, args.lowercase, args.inventory, has_header=(not args.no_header)) 
    
if __name__ == '__main__':
    main()
    