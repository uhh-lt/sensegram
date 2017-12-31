import argparse, sys, subprocess
from os.path import basename
import fast_top_nn.similar_top
import filter_clusters
import vector_representations.build_sense_vectors
from os.path import join
from utils.common import ensure_dir
import pcz

def get_paths(corpus_fpath, min_size):
    corpus_name = basename(corpus_fpath)
    model_dir = "model/"
    ensure_dir(model_dir)
    vectors_fpath = join(model_dir, corpus_name + ".words")
    neighbours_fpath = join(model_dir, corpus_name + ".neighbours")
    clusters_fpath = join(model_dir, corpus_name + ".clusters")
    clusters_minsize_fpath = clusters_fpath + ".minsize" + str(min_size) # clusters that satisfy min_size
    clusters_removed_fpath = clusters_minsize_fpath + ".removed" # cluster that are smaller than min_size

    return vectors_fpath, neighbours_fpath, clusters_fpath, clusters_minsize_fpath, clusters_removed_fpath

def stage1_learn_word_embeddings(corpus_fpath, vectors_fpath, cbow, window, iter_num, size, threads, min_count):

    bash_command = ("word2vec/bin/word2vec -train " +
                   corpus_fpath +
                   " -output " + vectors_fpath +
                   " -cbow " + str(cbow) +
                   " -size " + str(size) +
                   " -window " + str(window) +
                   " -threads " + str(threads) +
                   " -iter " + str(iter_num) +
                   " -min_count " + str(min_count) +
                   " -binary 0 -negative 25 -hs 0 -sample 1e-4")
    
    print("\n\n", "="*50, "\nSTAGE 1")
    print("Train word vectors using word2vec with following parameters:")
    print(bash_command)
    print("\nTraining progress won't be printed.")
    
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, ''):
        #pass
        sys.stdout.write(line.decode("utf-8"))


def stage2_compute_graph_of_related_words(vectors_fpath, neighbours_fpath, vocab_limit, only_letters, threads):
    print("\n\n", "="*50, "\nSTAGE 2")
    print("Start collection of word neighbours.")
    fast_top_nn.similar_top.run(vectors_fpath,
                                neighbours_fpath,
                                only_letters=only_letters,
                                vocab_limit=vocab_limit,
                                pairs=True,
                                batch_size=5000,
                                threads_num=threads,
                                word_freqs=None)


def stage3_graph_based_word_sense_induction(neighbours_fpath, clusters_fpath, clusters_minsize_fpath, clusters_removed_fpath, min_size, n, N):
    bash_command = ("java -Xms1G -Xmx130G -cp chinese-whispers/target/chinese-whispers.jar de.tudarmstadt.lt.wsi.WSI " +
                    " -in " + neighbours_fpath + " -out " + clusters_fpath +
                    " -N " + str(N) + " -n " + str(n) +
                    " -clustering cw")
    
    print("\n\n", "="*50, "\nSTAGE 3")
    print("\nStart clustering of word ego-networks with following parameters:")
    print(bash_command)
    
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line.decode("utf-8"))
    
    print("\nStart filtering of clusters.")
    
    filter_clusters.run(clusters_fpath, clusters_minsize_fpath, clusters_removed_fpath, str(min_size))


def stage4_building_sense_embeddings(clusters_minsize_fpath, vectors_fpath):
    print("\n\n", "="*50, "\nSTAGE 4")
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
    stage1_learn_word_embeddings(args.train_corpus, vectors_fpath, args.cbow, args.window,
                                 args.iter, args.size, args.threads, args.min_count)
    stage2_compute_graph_of_related_words(vectors_fpath, neighbours_fpath, args.vocab_limit, args.only_letters, args.threads)
    stage3_graph_based_word_sense_induction(neighbours_fpath, clusters_fpath, clusters_minsize_fpath, clusters_removed_fpath,
                                            args.min_size, args.n, args.N)
    stage4_building_sense_embeddings(clusters_minsize_fpath, vectors_fpath)

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
