import argparse, sys, subprocess
from os.path import basename
import word2vec_utils.similar_top 
import filter_clusters
import pooling



def init(args):
    global corpus
    global word_vectors
    global context_vectors
    global neighbours
    global clusters 
    global clusters_minsize # clusters that satisfy min_size
    global clusters_filtered # cluster that are smaller than min_size
    global sense_inventory
    global sense_vectors
    global sense_probabilities
    
    corpus = args.train_corpus
    corpus_name = basename(corpus)
    word_vectors = "model/" + corpus_name + ".words"
    context_vectors = "model/" + corpus_name + ".contexts"
    neighbours = "intermediate/" + corpus_name + ".neighbours"
    clusters = "intermediate/" + corpus_name + ".clusters"
    clusters_minsize = clusters + ".minsize" + unicode(args.min_size) # clusters that satisfy min_size
    clusters_filtered = clusters_minsize + ".filtered" # cluster that are smaller than min_size
    sense_inventory = "intermediate/" + corpus_name + ".inventory"
    sense_vectors = "model/" + corpus_name + ".senses.w2v"
    sense_probabilities = sense_vectors + ".probs"

def stage1(args):
    """ Train word vectors using word2vec """
    bash_command = ("word2vec_c/word2vec -train " + corpus + 
                   " -output " + word_vectors + " -save-ctx " + context_vectors +   
                   " -cbow " + unicode(args.cbow) + " -size " + unicode(args.size) + 
                   " -window " + unicode(args.window) + " -threads " + unicode(args.threads) +
                   " -iter " + unicode(args.iter) + " -min_count " + unicode(args.min_count) +
                   " -binary 1 -negative 25 -hs 0 -sample 1e-4")
    
    print "\nSTAGE 1"
    print "Start word vectors training with following parameters:"
    print bash_command
    
    print "\nTraining progress won't be printed."
    
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)

def stage2(args):
    print "\nSTAGE 2"
    print "Start collection of word neighbours."
    word2vec_utils.similar_top.init(word_vectors, neighbours, only_letters=args.only_letters, vocab_limit=args.vocab_limit, pairs=True, batch_size=1000, word_freqs=None)

def stage3(args):
    bash_command = ("java -Xms1G -Xmx2G -cp chinese-whispers/target/chinese-whispers.jar de.tudarmstadt.lt.wsi.WSI " +
                    " -in " + neighbours + " -out " + clusters + 
                    " -N " + unicode(args.N) + " -n " + unicode(args.n) +
                    " -clustering cw")
    
    print "\nSTAGE 3"
    print "\nStart clustering of word ego-networks with following parameters:"
    print bash_command
    
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
    
    print "\nStart filtering of clusters."
    
    filter_clusters.run(clusters, clusters_minsize, clusters_filtered, unicode(args.min_size))
    
def stage4(args):
    print "\nSTAGE 4"
    print "\nStart pooling of word vectors."
    
    pooling.run(clusters_minsize, word_vectors, sense_vectors, method=args.pooling_method, inventory=sense_inventory, lowercase=False, has_header=True)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('train_corpus', help="Path to training corpus")
    parser.add_argument('-cbow', help="Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)", default=1, type=int)
    parser.add_argument('-size', help="Set size of word vectors; default is 100", default=100, type=int)
    parser.add_argument('-window', help="Set max skip length between words; default is 3", default=3, type=int)
    parser.add_argument('-threads', help="Use <int> threads (default 12)", default=12, type=int)
    parser.add_argument('-iter', help="Run <int> training iterations (default 5)", default=5, type=int)
    parser.add_argument('-min_count', help="This will discard words that appear less than <int> times; default is 5", default=5, type=int)
    
    parser.add_argument('-only_letters', help="Use only words built from letters/dash/point for DT.", action="store_true")
    parser.add_argument("-vocab_limit", help="Use only <int> most frequent words from word vector model for DT. By default use all words.", default=None, type=int)
    
    parser.add_argument('-N', help="Number of nodes in each ego-network", default=200, type=int)
    parser.add_argument('-n', help="Maximum number of edges a node can have in the network", default=200, type=int)
    parser.add_argument('-min_size', help="Minimum size of the cluster", default=5, type=int)
    
    parser.add_argument('-pooling_method', help="Method for pooling of word vectors: 'mean' or 'weighted_mean'. Default='weighted_mean'", default="weighted_mean")
    
    args = parser.parse_args()
    
    init(args)
    stage1(args)
    stage2(args)
    stage3(args)
    stage4(args)
    
if __name__ == '__main__':
    main()