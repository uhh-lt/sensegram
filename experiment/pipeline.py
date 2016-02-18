#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" file in progress
    Execute all steps of the pipeline
"""
# help: 
# In sys.path[0] you have the path of your currently running script.
# os.getcwd() current working directory
# line = "ls -l"
# Popen(line.split(), cwd="..")

from subprocess import Popen, PIPE, call, check_output
import word_neighbours, filter_clusters

###### Train word vector model from corpora file <name>.txt ###### 
def train_word_vectors(prefix):
    name = prefix
    bash_command = ("word2vec_c/word2vec -train corpora/" + name + ".txt " + 
                   "-save-ctx model/" + name + "_context_vectors.bin -output model/" + name + "_word_vectors.bin " + 
                   "-cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 12 -binary 1 -iter 15")
    #p = call(bash_command.split())
    #p = check_output(bash_command.split()) # p has stdout, waits until process finished
    #print p
    #p = Popen(bash_command.split(), stdout=PIPE) # p.communicate()[0] has stdout, doesn't wait until finished
    #print "bla" # this is executed right after process start and long before process has finished
    #print p.communicate()[0]
    p = Popen(bash_command.split())
    p.wait()

    # don't use wait() and PIPE together

###### Collect word neighbours for model <name>_word_vectors.bin ######
def collect_word_neighbours(prefix):
    name = prefix
    word_neighbours.collect_neighbours("model/" + name + "_word_vectors.bin", 
                                        #"model/text8_vectors.bin",
                                       "intermediate/" + name + "_neighbours.txt")

###### Cluster word neighbours. The algorithm performs local clustering for each word. ######
###### Each cluster thus represents one sense of a word. ###### 
def cluster_word_neighbours(prefix):
    """
        -Xms    min (start) heap size
        -Xmx    max heap size 
        -in     input neighbours file in word1<TAB>neighbour1<TAB>similarity format
        -out    name of cluster output file (add .gz for compressed output)
        -n      max. number of edges to process for each similar word (word subgraph connectivity)
        -N      max. number of similar words to process for a given word (size of word subgraph to be clustered)
        -clustering     clustering algorithm to use: 'cw' or 'mcl'
        -e      min. edge weight
    """
    bash_command = ("time java -Xms2G -Xmx2G -cp chinese-whispers/target/chinese-whispers.jar " +
                    "de.tudarmstadt.lt.wsi.WSI -in intermediate/" + prefix + "_neighbours.txt -n 200 -N 200 " +
                    "-out intermediate/" + prefix + "_clusters.txt -clustering cw")
    p = Popen(bash_command.split())
    p.wait()

###### Filter out small clusters (might represent noise) ######
def postprocess_clusters(prefix):
    return filter_clusters.run("intermediate/" + prefix + "_clusters.txt")
    # time dt/postprocess.py -min_size 5 dt/clusters.txt

def pool_vectors():
    bash_command = "time ./pooling.py intermediate/test_clusters_minsize5.csv 3999 model/test_word_vectors.bin model/test_sense_vectors.bin -lowercase -inventory intermediate/test_inventory.csv"
    p = Popen(bash_command.split())
    p.wait()


###### Run pipeline ######

#train_word_vectors("test")
#collect_word_neighbours("test")
#cluster_word_neighbours("test")
#left_clusters, avg_number_senses = postprocess_clusters("test")
pool_vectors()

