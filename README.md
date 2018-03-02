# SenseGram

This repository contains implementation of a method that takes as an input a word embeddings, such as word2vec and splits different senses of the input words. For instance, the vector for the word "table" will be split into "table (data)" and "table (furniture)" as shown below.

Our method performs word sense induction and disambiguation based on sense embeddings. Sense inventory is induced from exhisting word embeddings via clustering of ego-networks of related words. Detailed description of the method is available in the original paper:

- [**Original paper**](http://aclweb.org/anthology/W/W16/W16-1620.pdf)
- [**Presentation**](docs/presentation.pdf)
- [**Poster**](docs/poster.pdf)

The picture below illustrates the main idea of the underlying approach: 

![ego](docs/table3.png)

If you use the method please cite the following paper:

```
@InProceedings{pelevina-EtAl:2016:RepL4NLP,
  author    = {Pelevina, Maria  and  Arefiev, Nikolay  and  Biemann, Chris  and  Panchenko, Alexander},
  title     = {Making Sense of Word Embeddings},
  booktitle = {Proceedings of the 1st Workshop on Representation Learning for NLP},
  month     = {August},
  year      = {2016},
  address   = {Berlin, Germany},
  publisher = {Association for Computational Linguistics},
  pages     = {174--183},
  url       = {http://anthology.aclweb.org/W16-1620}
}
```

## Installation

This project is implemented in Python 3. It makes use of the word2vec toolkit (via gensim), FAISS for computation of graphs of related words, and the Chinese Whispers graph clustering algorithm. We suggest using Ubuntu Linux 16.04 for computation of the models and using it on a computational server (ideally from 64Gb of RAM and 16 cores) as some stages are computational intensive. To install all dependencies on Ubuntu Linux 16.04 use the following commands:

```
git clone --recursive https://github.com/tudarmstadt-lt/sensegram.git
make install-ubuntu-16-04
```

Note that this command also will bring you an appropriate vesion of Python 3 via Anaconda. If you already have a properly configured recent version of Python 3 and/or running a system different from Ubuntu 16.04, use the command ``make install`` to install the dependencies. Note however, that in this case, you will also need to install manually binary dependencies required by FAISS yourself. 

<!-- 
## Testing a pre-trained model

To download the pre-trained models execute the following command:

```
make download
```

To test with word sense embeddings you can use a pretrained model (sense vectors and sense probabilities). These sense vectors were induced from English Wikipedia using word2vec similarities between words in ego-networks. Probabilities are stored in a separate file and are not strictly necessary (if absent, the model will assign equal probabilities for every sense). To load sense vectors:

```
$ python
>>> import sensegram
>>> sv = sensegram.SenseGram.load_word2vec_format(wiki.senses.w2v, binary=True)
```
Probabilities of senses will be loaded automatically if placed in the same folder as sense vectors and named according to the same scheme as our pretrained files.

To examine how many senses were learned for a word call `get_senses` funcion:

```
>>> sv.get_senses("table")
[('table#0', 0.40206185567), ('table#1', 0.59793814433)]
```
The function returns a list of sense names with probabilities for each sense. As one can see, our model has learned two senses for the word "table".

To understand which word sense is represented with a sense vector use `most_similar` function:

```
>>> sv.wv.most_similar("table#1")
[('pile#1', 0.9263191819190979),
 ('stool#1', 0.918972909450531),
 ('tray#0', 0.9099194407463074),
 ('basket#0', 0.9083326458930969),
 ('bowl#1', 0.905775249004364),
 ('bucket#0', 0.895959198474884),
 ('box#0', 0.8930465579032898),
 ('cage#0', 0.8916786909103394),
 ('saucer#3', 0.8904291391372681),
 ('mirror#1', 0.8880348205566406)]
```
For example, "table#1" represents the sense related to furniture.

To use our word sense disambiguation mechanism you also need word vectors or context vectors, depending on the dismabiguation strategy. Those word and context vectors should be trained on the same corpus as sense vectors. 
You can download word and context vectors pretrained on English Wikipedia here:  word vector ```wiki.words``` and context vectors ```wiki.contexts```.

Our WSD mechanism supports two disambiguation strategies: one based on word similarities (`sim`) and another based on word probabilities (`prob`). The first one requires word vectors to represent context words and the second one requires context vectors for the same purpose. In following we provide a disambiguation example using similarity strategy.

First, load word vectors using gensim library:

```
from gensim.models import word2vec
wv = word2vec.Word2Vec.load_word2vec_format(wiki.words, binary=True)
```

Then initialise the WSD object with sense and word vectors:

```
wsd_model = wsd.WSD(sv, wv, window=5, method='sim', filter_ctx=3)
```
The settings have the following meaning: it will extract at most `window`*2 words around the target word from the sentence as context and it will use only three most discriminative context words for disambiguation. 

Now you can disambiguate the word "table" in the sentence "They bought a table and chairs for kitchen" using `dis_text` function. As input it takes a sentence with space separated tokens, a target word, and start/end indices of the target word in the given sentence.

```
>>> wsd_model.dis_text("They bought a table and chairs for kitchen", "table", 14, 19)
(u'table#1', [0.15628162913257754, 0.54676466664654355])
```
It outputs its guess of the correct sense, as well as scores it assigned to all known senses during disambiguation. As one can see, it guessed the correct sense of the word "table" related to the furniture. The vector for this sense can be obtained from `sv["table#1"]`.

## Pretrained models
We provide several pretrained sense models accompanied by word and context vectors necessary for disambiguation. We have trained them on two corpora: English Wikipedia dump and ukWaC.

#### English Wikipedia

The vectors are of size 300, trained with CBOW model using 3-words context window, 3 iterations and minimum word frequency of 5.

- Word and context vectors: ```wiki.words, wiki.contexts```
- Senses and probabilities induced using word2vec similarities between words: ```wiki.senses.w2v, wiki.senses.w2v.probs```
- Senses and probabilities induced using JoBimText similarities between words: ```wiki.senses.jbt, wiki.senses.jbt.probs```
- Senses and probabilities based on TWSI sense inventory: ```wiki.senses.twsi, wiki.senses.twsi.probs```

#### ukWaC

The vectors are of size 100, trained with CBOW model using 3-words context window, 3 iterations and minimum word frequency of 5.

- Word and context vectors: ```ukwac.words, ukwac.contexts```
- Senses and probabilities induced using word2vec similarities between words: ```ukwac.senses.w2v, ukwac.senses.w2v.probs```
- Senses and probabilities induced using JoBimText similarities between words: ```ukwac.senses.jbt, ukwac.senses.jbt.probs```
- Senses and probabilities based on TWSI sense inventory: ```ukwac.senses.twsi, ukwac.senses.twsi.probs```

-->

## Training a new model from a text corpus

The way to train your own sense embeddings is with the `train.py` script. You will have to provide a raw text corpus as input. If you run `train.py` with no parameters, it will print usage information:

```
usage: train.py [-h] [-cbow CBOW] [-size SIZE] [-window WINDOW]
                [-threads THREADS] [-iter ITER] [-min_count MIN_COUNT] [-N N]
                [-n N] [-min_size MIN_SIZE] [-make-pcz]
                train_corpus

Performs training of a word sense embeddings model from a raw text corpus
using the SkipGram approach based on word2vec and graph clustering of ego
networks of semantically related terms.

positional arguments:
  train_corpus          Path to a training corpus in text form (can be .gz).

optional arguments:
  -h, --help            show this help message and exit
  -cbow CBOW            Use the continuous bag of words model (default is 1,
                        use 0 for the skip-gram model).
  -size SIZE            Set size of word vectors (default is 300).
  -window WINDOW        Set max skip length between words (default is 5).
  -threads THREADS      Use <int> threads (default 40).
  -iter ITER            Run <int> training iterations (default 5).
  -min_count MIN_COUNT  This will discard words that appear less than <int>
                        times (default is 10).
  -N N                  Number of nodes in each ego-network (default is 200).
  -n N                  Maximum number of edges a node can have in the network
                        (default is 200).
  -min_size MIN_SIZE    Minimum size of the cluster (default is 5).
  -make-pcz             Perform two extra steps to label the original sense
                        inventory with hypernymy labels and disambiguate the
                        list of related words.The obtained resource is called
                        proto-concepualization or PCZ.
```

The training produces following output files:

* `model/ + CORPUS_NAME + .word_vectors` - word vectors
* `model/ + CORPUS_NAME + .sense_vectors` - sense vectors
* `model/ + CORPUS_NAME + .sense_vectors.inventory.csv` - sense probabilities  

In addition, it produces several intermediary files that can be investigated for error analysis or removed after training:

* `model/ + CORPUS_NAME + .graph` - word similarity graph (distributional thesaurus) 
* `model/ + corpus_name + .clusters` - sense clusters produced by chinese-whispers
* `model/ + corpus_name + .minsize + MIN_SIZE` - clusters that remained after filtering out of small clusters 

In [train.sh](train.sh) we provide an example for usage of the `train.py` script. You can test it using the command ``make train``. More useful commands can be found in the [Makefile](Makefile).

## Using a pre-trained model 

See the [QuickStart](QuickStart.ipynb) tutorial on how to perform word sense disambiguation and inspection of a trained model.

You can downlooad [pre-trained models for English, German, and Russian](http://ltdata1.informatik.uni-hamburg.de/sensegram/). Note that to run examples from the QuickStart you only need files with extensions ``.word_vectors``, ``.sense_vectors``, and ``.sense_vectors.inventory.csv``. Other files are supplementary.

## Transforming pre-trained word embeddings to sense embeddings

Instead of learning a model from a text corpus, you can provide a pre-trained word embedding model. To do so, you just neeed to:

1. Save the word embeddings file (in word2vec text format) with the extension ``.word_vectors``, e.g. ``wikipedia.word_vectors``. 

2. Run the ``train.py`` script inducating the path to the word embeddings file, e.g.:

```python train.py model/wikipedia```

Note: do not indicate the ``.word_vectors`` extension when launching the train.py script. 
