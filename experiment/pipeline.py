""" file in progress
    Execute all steps of the pipeline
"""
# help: 
# In sys.path[0] you have the path of your currently running script.
# os.getcwd() current working directory
# line = "ls -l"
# Popen(line.split(), cwd="..")
import word_neighbours
from subprocess import Popen, PIPE, call, check_output

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

# to apply clustering:
# cd ../chinese-whispers/           
# time java -Xms2G -Xmx2G -cp target/chinese-whispers.jar de.tudarmstadt.lt.wsi.WSI -in ../experiment/dt/neighbours.txt -n 200 -N 200 -out ../experiment/dt/clusters.txt -clustering cw -e 0.01
#  -n <integer>           max. number of edges to process for each similar
#                         word (word subgraph connectivity)
#  -N <integer>           max. number of similar words to process for a
#                         given word (size of word subgraph to be clustered)

# to postprocess clusters:
# cd ../experiment
# time dt/postprocess.py -min_size 5 dt/clusters.txt

###### Start pipeline ######

#train_word_vectors("test")
collect_word_neighbours("test")
