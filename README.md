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

## Use cases

This software can be used to:

- Generation of word sense embeddigns from a raw text corpus

- Generation of word sense embeddings from a pretrained word embeddings (in the word2vec format)

- Generation of graphs of semantically related words

- Generation of graphs of semantically related word senses

- Generation of a word sense inventory specific to the input text corpus



## Installation

This project is implemented in Python 3. It makes use of the word2vec toolkit (via gensim), FAISS for computation of graphs of related words, and the Chinese Whispers graph clustering algorithm. We suggest using Ubuntu Linux 16.04 for computation of the models and using it on a computational server (ideally from 64Gb of RAM and 16 cores) as some stages are computational intensive. To install all dependencies on Ubuntu Linux 16.04 use the following commands:

```
git clone --recursive https://github.com/tudarmstadt-lt/sensegram.git
make install-ubuntu-16-04
```

Optional: Set the ``PYTHONPATH`` variable to the root directory of this repository (needed only for working with the "egvi" scripts), e.g. 
``export PYTHONPATH="/home/user/sensegram:$PYTHONPATH"

Note that this command also will bring you an appropriate vesion of Python 3 via Anaconda. If you already have a properly configured recent version of Python 3 and/or running a system different from Ubuntu 16.04, use the command ``make install`` to install the dependencies. Note however, that in this case, you will also need to install manually binary dependencies required by FAISS yourself. 


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

* `model/ + CORPUS_NAME + .word_vectors` - word vectors in the word2vec text format
* `model/ + CORPUS_NAME + .sense_vectors` - sense vectors in the word2vec text format
* `model/ + CORPUS_NAME + .sense_vectors.inventory.csv` - sense probabilities in TSV format

In addition, it produces several intermediary files that can be investigated for error analysis or removed after training:

* `model/ + CORPUS_NAME + .graph` - word similarity graph (distributional thesaurus) in TSV format
* `model/ + corpus_name + .clusters` - sense clusters produced by chinese-whispers in TSV format
* `model/ + corpus_name + .minsize + MIN_SIZE` - clusters that remained after filtering out of small clusters in TSV format

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
