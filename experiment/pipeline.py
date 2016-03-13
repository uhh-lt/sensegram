#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Execute all steps of the pipeline for a test model.
    You can use python modules either as executable scripts or as modules to import.
"""
# help: 
# In sys.path[0] you have the path of your currently running script.
# os.getcwd() current working directory
# line = "ls -l"
# Popen(line.split(), cwd="..")

from subprocess import Popen, PIPE, call, check_output
import word_neighbours, filter_clusters

#######################                                 ###########################
####################### Train word/context vector model ###########################
def train_word_vectors():
    bash_command = ("word2vec_c/word2vec -train corpora/test.txt -save-ctx model/test_context_vectors.bin -output model/test_word_vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 12 -binary 1 -iter 15")
    #p = call(bash_command.split())
    #p = check_output(bash_command.split()) # p has stdout, waits until process finished
    #print p
    #p = Popen(bash_command.split(), stdout=PIPE) # p.communicate()[0] has stdout, doesn't wait until finished
    #print "bla" # this is executed right after process start and long before process has finished
    #print p.communicate()[0]
    p = Popen(bash_command.split())
    p.wait()

    # don't use wait() and PIPE together

#######################                      ###########################
####################### Induce sense vectors ###########################

###### Collect word neighbours ######
def collect_word_neighbours():
    word_neighbours.run("model/test_word_vectors.bin", 
                                       "intermediate/test_neighbours.txt")
    # ./word_neighbours.py model/test_word_vectors.bin intermediate/test_neighbours.txt -n 200

###### Cluster word neighbours. The algorithm performs local clustering for each word. ######
###### Each cluster thus represents one sense of a word. ###### 
def cluster_word_neighbours():
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
    bash_command = ("time java -Xms2G -Xmx2G -cp chinese-whispers/target/chinese-whispers.jar de.tudarmstadt.lt.wsi.WSI -in intermediate/test_neighbours.txt -n 200 -N 200 -out intermediate/test_clusters.txt -clustering cw")
    p = Popen(bash_command.split())
    p.wait()

###### Filter out small clusters (might represent noise) ######
def postprocess_clusters():
    return filter_clusters.run("intermediate/test_clusters.txt")
    # time ./filter_clusters.py intermediate/test_clusters.txt -min_size 5

###### Create sense vectors ######
def pool_vectors():
    bash_command = "time ./pooling.py intermediate/test_clusters_minsize5.csv 3999 model/test_word_vectors.bin model/test_sense_vectors.bin -lowercase -inventory intermediate/test_inventory.csv"
    p = Popen(bash_command.split())
    p.wait()

#######################                      ###########################
####################### Tests and evaluation ###########################

###### Sanity check: disambiguate occurences of words in the initial corporus. Observe different parameters. ######
def put_back():
    bash_command = "time ./put_back.py model/test_sense_vectors.bin model/test_context_vectors.bin corpora/test.txt eval/test_put_back.txt -lowercase -words anarchism,estate"
    p = Popen(bash_command.split())
    p.wait()

###### Fill in a test set for WSD ######
def predict():

    # SemEval dataset 
    bash_command = "time ./prediction.py context-eval/data/Dataset-SemEval-2013-13.csv model/test_sense_vectors.bin model/test_context_vectors.bin eval/test_SemEval-2013-13_predictions_nothr.csv -lowercase"
    p = Popen(bash_command.split())
    p.wait()

    # TWSI dataset
    bash_command = "time ./prediction.py context-eval/data/Dataset-TWSI-2.csv model/test_sense_vectors.bin model/test_context_vectors.bin eval/test_TWSI-2_predictions_nothr.csv -lowercase"
    p = Popen(bash_command.split())
    p.wait()

###### Evaluate testsets ######
    # TODO: write correctly
    #(cd context-eval/ && exec time ./semeval_2013_13.sh semeval_2013_13/keys/gold/all.key ../eval/test_SemEval-2013-13_predictions_nothr.csv > ../eval/corpus_en.norm-sz100-w1-cb0-it1-min20_SemEval-2013-13_predictions_nothr.csv.eval)
    #(cd context-eval/ && exec time ./twsi_evaluation.py ../intermediate/test_inventory.csv ../eval/test_TWSI-2_predictions_nothr.csv)

#######################                     ###########################
#######################    Run pipeline     ###########################

#train_word_vectors()
#collect_word_neighbours()
#cluster_word_neighbours()
#left_clusters, avg_number_senses = postprocess_clusters()
#pool_vectors()
#put_back()
predict()

