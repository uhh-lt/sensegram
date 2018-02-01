import argparse
from utils.common import exists
from os.path import basename
import gensim 
from time import time 
from os.path import join

import filter_clusters
import vector_representations.build_sense_vectors
from utils.common import ensure_dir
import pcz
from word_embeddings import learn_word_embeddings
from word_sense_induction import ego_network_clustering


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
