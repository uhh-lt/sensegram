import argparse, sys, subprocess
from utils.common import exists
from os.path import basename
import gensim 
import gzip
from gensim.models.keyedvectors import KeyedVectors
from gensim.utils import tokenize
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from time import time 
import numpy as np
from chinese_whispers import chinese_whispers, aggregate_clusters
import codecs 
import networkx as nx
from multiprocessing import Pool 
from os.path import join
import faiss

import filter_clusters
import vector_representations.build_sense_vectors
from utils.common import ensure_dir
import pcz


verbose = False


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


def get_paths(corpus_fpath, min_size):
    corpus_name = basename(corpus_fpath)
    model_dir = "model/"
    ensure_dir(model_dir)
    vectors_fpath = join(model_dir, corpus_name + ".vectors")
    neighbours_fpath = join(model_dir, corpus_name + ".graph")
    clusters_fpath = join(model_dir, corpus_name + ".clusters")
    clusters_minsize_fpath = clusters_fpath + ".minsize" + str(min_size) # clusters that satisfy min_size
    clusters_removed_fpath = clusters_minsize_fpath + ".removed" # cluster that are smaller than min_size

    return vectors_fpath, neighbours_fpath, clusters_fpath, clusters_minsize_fpath, clusters_removed_fpath


def get_clustered_ego_network(ego):
    tic = time()
    ego_network = nx.Graph(name=ego)

    # Add related and substring nodes 
    substring_nodes = []
    for j, node in enumerate(G.nodes):
        if ego.lower() == node.lower():
            ego_network.add_nodes_from( [(rn, {"weight": G[node][rn]["weight"]})
                                         for rn in G[node].keys()] )
        else:
            if "_" not in node: continue
            if node.startswith(ego + "_") or node.endswith("_" + ego):
                
                if ego in G and node in G[ego]: w = G[ego][node]["weight"]
                else: w = 0.99
                    
                substring_nodes.append( (node, {"weight": w}) )
                
    ego_network.add_nodes_from(substring_nodes)

    # Find edges of the ego network
    for r_node in ego_network:
        related_related_nodes = G[r_node]
        related_related_nodes_ego = sorted(
            [(related_related_nodes[rr_node]["weight"], rr_node) for rr_node in related_related_nodes if rr_node in ego_network],
            reverse=True)[:n]
        related_edges = [(r_node, rr_node, {"weight": w}) for w, rr_node in  related_related_nodes_ego]

        ego_network.add_edges_from(related_edges)

    # Perform clustering   
    chinese_whispers(ego_network, weighting="top", iterations=20)
    if verbose: print("{}\t{:f} sec.".format(ego, time()-tic))
        
    return ego_network

G = None
n = None

def ego_network_clustering(neighbors_fpath, clusters_fpath, max_related=300, num_cores=32): 
    global G
    global n
    n = max_related
    G = nx.read_edgelist(neighbors_fpath, nodetype=str, delimiter="\t", data=(('weight',float),))

    with codecs.open(clusters_fpath, "w", "utf-8") as output, Pool(num_cores) as pool:    
        output.write("word\tcid\tcluster\tisas\n") 

        for i, ego_network in enumerate(pool.imap_unordered(get_clustered_ego_network, G.nodes)): 
            if i % 50000 == 0: print(i, "word processed")
            sense_num = 1
            for label, cluster in sorted(aggregate_clusters(ego_network).items(), key=lambda e: len(e[1]), reverse=True):
                output.write("{}\t{}\t{}\t\n".format(
                    ego_network.name,
                    sense_num,
                    ", ".join( ["{}:{:.4f}".format(c_node, ego_network.node[c_node]["weight"]) for c_node in cluster] )
                ))
                sense_num += 1

        print("Clusters:", clusters_fpath)


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

 
def compute_graph_of_related_words(vectors_fpath, neighbours_fpath, neighbors=200):
    print("Start collection of word neighbours.")
    tic = time()
    index, w2v = build_vector_index(vectors_fpath)
    compute_neighbours(index, w2v, neighbours_fpath, neighbors)
    print("Elapsed: {:f} sec.".format(time() - tic))


def word_sense_induction(neighbours_fpath, clusters_fpath, n, threads):
    print("\nStart clustering of word ego-networks.")
    tic = time()
    ego_network_clustering(neighbours_fpath, clusters_fpath, max_related=n, num_cores=threads)
    print("Elapsed: {:f} sec.".format(time() - tic))


def building_sense_embeddings(clusters_minsize_fpath, vectors_fpath):
    print("\nStart pooling of word vectors.")
    vector_representations.build_sense_vectors.run(
        clusters_minsize_fpath, vectors_fpath, sparse=False,
        norm_type="sum", weight_type="score", max_cluster_words=20)


def main():
    parser = argparse.ArgumentParser(description='Performs training of a word sense embeddings model from a raw text '
                                                 'corpus using the SkipGram approach based on word2vec and graph '
                                                 'clustering of ego networks of semantically related terms.')
    parser.add_argument('train_corpus', help="Path to a training corpus.")
    parser.add_argument('-cbow', help="Use the continuous bag of words model (default is 1, use 0 for the "
                                      "skip-gram model).", default=1, type=int)
    parser.add_argument('-size', help="Set size of word vectors (default is 300).", default=300, type=int)
    parser.add_argument('-window', help="Set max skip length between words (default is 5).", default=5, type=int)
    parser.add_argument('-threads', help="Use <int> threads (default 4).", default=4, type=int)
    parser.add_argument('-iter', help="Run <int> training iterations (default 5).", default=5, type=int)
    parser.add_argument('-min_count', help="This will discard words that appear less than <int> times"
                                           " (default is 5).", default=5, type=int)
    parser.add_argument('-only_letters', help="Use only words built from letters/dash/point for DT.", action="store_true")
    parser.add_argument('-vocab_limit', help="Use only <int> most frequent words from word vector model"
                                             " for DT. By default use all words (default is none).", default=None, type=int)
    parser.add_argument('-N', help="Number of nodes in each ego-network (default is 200).", default=200, type=int)
    parser.add_argument('-n', help="Maximum number of edges a node can have in the network"
                                   " (default is 200).", default=200, type=int)
    parser.add_argument('-min_size', help="Minimum size of the cluster (default is 5).", default=5, type=int)
    parser.add_argument('-make-pcz', help="Perform two extra steps to label the original sense inventory with"
                                          " hypernymy labels and disambiguate the list of related words."
                                          "The obtained resource is called proto-concepualization or PCZ.", action="store_true")
    args = parser.parse_args()

    vectors_fpath, neighbours_fpath, clusters_fpath, clusters_minsize_fpath, clusters_removed_fpath = get_paths(
        args.train_corpus, args.min_size)
    
    if not exists(vectors_fpath):
        learn_word_embeddings(args.train_corpus, vectors_fpath, args.cbow, args.window,
                              args.iter, args.size, args.threads, args.min_count)
    else:
        print("Using existing vectors:", vectors_fpath)
 
    if not exists(neighbours_fpath):
        compute_graph_of_related_words(vectors_fpath, neighbours_fpath)
    else:
        print("Using existing neighbors:", neighbours_fpath)
        
    if not exists(clusters_fpath):
        word_sense_induction(neighbours_fpath, clusters_fpath, args.n, args.threads)
    else:
       print("Using existing clusters:", clusters_fpath)
   
    if not exists(clusters_minsize_fpath): 
        filter_clusters.run(clusters_fpath, clusters_minsize_fpath, args.min_size)
    else:
        print("Using existing filtered clusters:", clusters_minsize_fpath)
    
    building_sense_embeddings(clusters_minsize_fpath, vectors_fpath)

    if (args.make_pcz):
        # add isas
        isas_fpath = ""
        # in: clusters_minsize_fpath
        clusters_with_isas_fpath = clusters_minsize_fpath + ".isas"


        # disambiguate the original sense clusters
        clusters_disambiguated_fpath = clusters_with_isas_fpath + ".disambiguated"
        pcz.disamgiguate_sense_clusters.run(clusters_with_isas_fpath, clusters_disambiguated_fpath)

        # make the closure
        clusters_closure_fpath = clusters_disambiguated_fpath + ".closure"
        # in: clusters_disambiguated_fpath


if __name__ == '__main__':
    main()
