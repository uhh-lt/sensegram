Scripts for the experiment described here: https://github.com/tudarmstadt-lt/sensegram/issues/2

##1. word_neighbours.py
Input: a pre-trained model of word vectors (use word2vec or gensim) 
Loads the model and outputs 200 nearest neighbours for every word in the vocabulary in the DT format.
```
word_j<TAB>word_j<TAB>similarity_ij
```
Output: dt/neighbours.txt 


##2. cluster.sh 
Input: dt/neighbours.txt. 
Calls java chinese-whispers clustering (https://github.com/tudarmstadt-lt/chinese-whispers).
Inidicate the path to your own chinese-whispers.jar or place the chinese-whispers folder next to the experiments foder.
Output: dt/clusters.txt - clusters of neighbours of each word.

##3. dt/postprocess.py
Input: dt/clusters.txt
Filters out any cluster with less than 5 elements.
Output: dt/clusters-minsize5.csv all cluster with 5 elements or more.
		dt/clusters-minsize5-filtered.csv all clusters with less than 5 elements.
		
##4. word_sense_pooling.py
Input: dt/clusters-minsize5.csv
Calculated the mean of vectors of all words in one cluster. Such mean is a new sense vector.
Output: model/text8_sense_vectors.bin

##5. create_sense_inventory.py
Input: model/wiki_sense_vectors.bin.
Creates an inventory as required for contextualization evaluation (https://github.com/tudarmstadt-lt/contextualization-eval#input-data-format-datadataset-twsi-20csv).
Format of inventory: 
```
Word<Tab>SenseID<Tab>list:5,of:3,related:1,words:1
```
Realted words are defined as 50 nearest neighbours of each sense.
Output: dt/inventory.csv - the inventory, dt/sanity_check.txt 50 nearest neighbours in a different format (see step 6 of experiment description).
		
##6. prediction.py
Input: model/wiki_sense_vectors.bin , model/wiki_vectors.bin.contexts , contextualization-eval/data/predictions.csv. 
Runs the disambiguation model (defined in wsd.py) over the TWSI test set (see https://github.com/tudarmstadt-lt/contextualization-eval#input-data-format-datadataset-twsi-20csv).
Output: predictions.csv - file in the format:
```
'context_id\ttarget\ttarget_pos\ttarget_position\tgold_sense_ids\tpredict_sense_ids\tgolden_related\tpredict_related\tcontext'
```

##7. contextualization-eval/twsi_evaluation.py
Input predictions.csv, inventory.csv
Evaluates predictions from previous step. This script is different from distributed version (minor bugs corrected). Run:
``` 
cd contextualization-eval/
time python twsi_evaluation.py ../dt/inventory.csv data/predictions.csv 

# Results
All datasets and models can be found on frink server:
```
home/pelevina/experiment/
```

Among most useful files are:
* model/wiki_sense_vectors.bin - sense vectors
* model/wiki_vectors.bin.contexts - context vectors
* dt/inventory.csv - 100 nearest neighbours for every sense vector


		
Contents of `dt/` and `model/` folders will be uploaded separately (to frink).
