import argparse, sys, subprocess
from os.path import basename
import word2vec_utils.similar_top 
import filter_clusters
import os
import jnt.verbs.build_sense_vectors
from os.path import join

def ensure_dir(f):
    """ Make the directory. """
    if not os.path.exists(f): os.makedirs(f)


corpus_fpath = ""
vectors_fpath = ""
neighbours_fpath = ""
clusters_fpath  = ""
clusters_minsize_fpath = ""
clusters_removed_fpath = ""

def init(args):
    global corpus_fpath
    global vectors_fpath
    global neighbours_fpath
    global clusters_fpath
    global clusters_minsize_fpath
    global clusters_removed_fpath

    corpus_fpath = args.train_corpus
    corpus_name = basename(corpus_fpath)
    model_dir = "model/"
    ensure_dir(model_dir)
    vectors_fpath = join(model_dir, corpus_name + ".words")
    neighbours_fpath = join(model_dir, corpus_name + ".neighbours")
    clusters_fpath = join(model_dir, corpus_name + ".clusters")
    clusters_minsize_fpath = clusters_fpath + ".minsize" + unicode(args.min_size) # clusters that satisfy min_size
    clusters_removed_fpath = clusters_minsize_fpath + ".removed" # cluster that are smaller than min_size


def stage1_learn_word_embeddings(args):
    """ Train word vectors using word2vec """
    bash_command = ("word2vec/bin/word2vec -train " + corpus_fpath +
                   " -output " + vectors_fpath +
                   " -cbow " + unicode(args.cbow) + " -size " + unicode(args.size) + 
                   " -window " + unicode(args.window) + " -threads " + unicode(args.threads) +
                   " -iter " + unicode(args.iter) + " -min_count " + unicode(args.min_count) +
                   " -binary 0 -negative 25 -hs 0 -sample 1e-4")
    
    print "\n\n", "="*50, "\nSTAGE 1"
    print "Start word vectors training with following parameters:"
    print bash_command
    
    print "\nTraining progress won't be printed."
    
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)

def stage2_compute_graph_of_related_words(args):
    print "\n\n", "="*50, "\nSTAGE 2"
    print "Start collection of word neighbours."
    word2vec_utils.similar_top.run(vectors_fpath, neighbours_fpath, only_letters=args.only_letters,
                                   vocab_limit=args.vocab_limit, pairs=True, batch_size=5000,
                                   threads_num=args.threads, word_freqs=None)

def stage3_graph_based_word_sense_induction(args):
    bash_command = ("java -Xms1G -Xmx130G -cp chinese-whispers/target/chinese-whispers.jar de.tudarmstadt.lt.wsi.WSI " +
                    " -in " + neighbours_fpath + " -out " + clusters_fpath +
                    " -N " + unicode(args.N) + " -n " + unicode(args.n) +
                    " -clustering cw")
    
    print "\n\n", "="*50, "\nSTAGE 3"
    print "\nStart clustering of word ego-networks with following parameters:"
    print bash_command
    
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
    
    print "\nStart filtering of clusters."
    
    filter_clusters.run(clusters_fpath, clusters_minsize_fpath, clusters_removed_fpath, unicode(args.min_size))
    
def stage4_building_sense_embeddings(args):
    print "\n\n", "="*50, "\nSTAGE 4"
    print "\nStart pooling of word vectors."
    jnt.verbs.build_sense_vectors.run(clusters_minsize_fpath, vectors_fpath, sparse=False, norm_type="sum", weight_type="score",
                                      max_cluster_words=20)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('train_corpus', help="Path to training corpus")
    parser.add_argument('-cbow', help="Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)", default=1, type=int)
    parser.add_argument('-size', help="Set size of word vectors; default is 300", default=300, type=int)
    parser.add_argument('-window', help="Set max skip length between words; default is 5", default=5, type=int)
    parser.add_argument('-threads', help="Use <int> threads (default 4)", default=4, type=int)
    parser.add_argument('-iter', help="Run <int> training iterations (default 5)", default=5, type=int)
    parser.add_argument('-min_count', help="This will discard words that appear less than <int> times; default is 5", default=5, type=int)
    parser.add_argument('-only_letters', help="Use only words built from letters/dash/point for DT.", action="store_true")
    parser.add_argument("-vocab_limit", help="Use only <int> most frequent words from word vector model for DT. By default use all words.", default=None, type=int)
    parser.add_argument('-N', help="Number of nodes in each ego-network", default=200, type=int)
    parser.add_argument('-n', help="Maximum number of edges a node can have in the network", default=200, type=int)
    parser.add_argument('-min_size', help="Minimum size of the cluster", default=5, type=int)
    args = parser.parse_args()
    
    init(args)
    stage1_learn_word_embeddings(args)
    stage2_compute_graph_of_related_words(args)
    stage3_graph_based_word_sense_induction(args)
    stage4_building_sense_embeddings(args)
    
if __name__ == '__main__':
    main()
