# SenseGram
A project for sense embeddings induction and subsequent evaluation.
## How to use
We will guide you through all steps from vectors training to the evaluation with this toy *test* example. 

##### Step 0: Train models
To begin you will need a word vector model and a corresponding context vector model. If you don't have them, you can train these models from a corpus of your choice using word2vec:

```sh
word2vec_c/word2vec -train corpora/test.txt -save-ctx model/test_context_vectors.bin -output model/test_word_vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 12 -binary 1 -iter 15
```
This is a [adaptation][word2vec_c] of original word2vec implementation. It's option -save_ctx  allows to save context vectors along with word vectors. Please, refer to [word2vec] documentation for information about word vectors in general and other training options.

##### Step 1: Build sense vectors
At first we collect 200 nearest neighbours for each word in the vocabulary of the model:
```sh
./word_neighbours.py model/test_word_vectors.bin intermediate/test_neighbours.txt -n 200
```
Then we cluster neighbours of each word using [Chinese Whispers] algorithm:
```sh
java -Xms2G -Xmx2G -cp chinese-whispers/target/chinese-whispers.jar de.tudarmstadt.lt.wsi.WSI -in intermediate/test_neighbours.txt -n 200 -N 200 -out intermediate/test_clusters.txt -clustering cw
```
and filter out clusters with less than 5 elements, because they are most likely to introduce noise into the model.
```sh
./filter_clusters.py intermediate/test_clusters.txt -min_size 5
```
File with clusters after filtering is placed next to the original `intermediate/test_clusters.txt` under the same name with a suffix `_minsize5.csv`. According to intuition, each cluster represents a sense of a word.
At last we build a sense vector for every cluster by pooling word vectors of words belonging to this cluster:
```sh
./pooling.py intermediate/test_clusters_minsize5.csv 3999 model/test_word_vectors.bin model/test_sense_vectors.bin -method mean -lowercase -inventory intermediate/test_inventory.csv
```
Number 3999 specifies how many clusters remained after filtering. Currently the vectors are pooled by averaging (`-method mean`). If `-inventory` parameter is specified, this script also outputs a sense inventory for the built sense vector model.

##### Step 2: Evaluate sense embeddings
For evaluation we use [context-eval] tools. We fill in the TWSI and SemEval test sets with our sense predictions using sense and context vectors for disambiguiation:
```sh
./prediction.py context-eval/data/Dataset-SemEval-2013-13.csv model/test_sense_vectors.bin model/test_context_vectors.bin eval/test_SemEval-2013-13_predictions_nothr.csv -lowercase

./prediction.py context-eval/data/Dataset-TWSI-2.csv model/test_sense_vectors.bin model/test_context_vectors.bin eval/test_TWSI-2_predictions_nothr.csv -lowercase
```
After that we can evaluate our predictions:
```
cd context-eval
./semeval_2013_13.sh semeval_2013_13/keys/gold/all.key ../eval/test_SemEval-2013-13_predictions_nothr.csv

./twsi_evaluation.py ../intermediate/test_inventory.csv ../eval/test_TWSI-2_predictions_nothr.csv
cd ..
```
Results of evaluation are printed to stdout.

**Note:** This example assumes that implementations of word2vec_c, chinese-whispers and context-eval reside in the root of this project in directories of the same names.
**Note:** on frink:/home/pelevina/experiment you can find further model and evaluation results

## Description of files and folders
Apart from python files mentioned above the project contains the following:
* wsd.py -- implementation of WSD class which represents a disambiguation model. It is initialized with sense and context vectors and provides functions to disambiguate a word in given context.
* put_back.py -- a sanity check. Finds occurences of specified words in the original corpus and disambiguates them. Example call:

        ./put_back.py model/test_sense_vectors.bin model/test_context_vectors.bin corpora/test.txt eval/test_put_back.txt -lowercase -words anarchism,estate
        
* pbar.py -- custom implementation of a progress bar for loops.
* runs/ -- further usage examples & definitions of jobs to be run on [cluster]
* corpora/ -- by convention use this folder to store initial corpora used to train word/context vectors.
* model/ -- by convention use this folder to store all types of vector models
* intermediate/ -- by convention use this folder to store neighbours, clusters and inventories
* eval/ -- by convention use this folder to store filled test sets and evaluation resuls

[word2vec]:https://code.google.com/archive/p/word2vec/
[word2vec_c]:https://github.com/tudarmstadt-lt/sensegram/tree/master/word2vec_c
[Chinese Whispers]:https://github.com/tudarmstadt-lt/chinese-whispers
[context-eval]:https://github.com/tudarmstadt-lt/context-eval
[cluster]:http://www.hhlr.tu-darmstadt.de/hhlr/index.de.jsp


