# SenseGram
A system for word sense induction and disambiguation based on sense embeddings. Sense inventory is induced from exhisting word embeddings via clustering of ego-networks of related words.

## Requirements
This project is implemented in Python 2.7. 
It makes use of a modified implementation of word2vec toolkit that saves context vectors ([word2vec_c](word2vec_c/)) and a clustering algorithm [chinese_whispers](chinese_whispers/), both distributed with this package. In addition to this it requires an installation of [gensim](https://pypi.python.org/pypi/gensim) library that provides a python implementation of word2vec toolkit.

## Playing with the model
To play with word sense embeddings you can use a pretrained model (sense vectors and sense probabilities). These sense vectors were induced from English Wikipedia using word2vec similarities between words in ego-networks. Probabilities are stored in a separate file and are not strictly necessary (if absent, the model will assign equal probabilities for every sense). To download the model call:

```
wget /home/pelevina/experiment/model/public/wiki.senses.w2v
wget /home/pelevina/experiment/model/public/wiki.senses.w2v.probs
```

To load sense vectors:

```
$ python
>>> import sense2vec
>>> sv = sense2vec.Sense2Vec.load_word2vec_format(PATH_TO_SENSE_VECTORS, binary=True)
```
Probabilities of senses will be loaded automatically if placed in the same folder as sense vectors and named according to the same scheme as our pretrained files.

To examine how many senses were learned for a word call `get_senses` funcion:

```
>>> sv.get_senses("table")
[(table#0', 0.40206185567), (table#1', 0.59793814433)]
```
The function returnes a list of sense names with probabilities for each sense. As one can see, our model has learned two senses for the word "table".

To understand which word sense is represented with a sense vector use `most_similar` function:

```
>>> sv.most_similar("table#1")
[(u'pile#1', 0.9263191819190979),
 (u'stool#1', 0.918972909450531),
 (u'tray#0', 0.9099194407463074),
 (u'basket#0', 0.9083326458930969),
 (u'bowl#1', 0.905775249004364),
 (u'bucket#0', 0.895959198474884),
 (u'box#0', 0.8930465579032898),
 (u'cage#0', 0.8916786909103394),
 (u'saucer#3', 0.8904291391372681),
 (u'mirror#1', 0.8880348205566406)]
```
For example, "table#1" represents the sense related to furniture.

To use our word sense disambiguation mechanism you also need word vectors or context vectors, depending on the dismabiguation strategy. Those word and sense vectors should be trained on the same corpus with a modified word2vec implementation. You can download word and context vectors pretrained on English Wikipedia here:

```
wget 130.83.164.196/home/pelevina/experiment/model/public/wiki.words
wget 130.83.164.196/home/pelevina/experiment/model/public/wiki.contexts
```

Our WSD mechanism supports two disambiguation strategies: one based on word similarities (`sim`) and on word probabilities (`prob`). The first one requires word vectors to represent context words and the second one requires context vectors for the same purpose. In following we provide a disambiguation example using similarity strategy.

First, load word vectors using gensim library:

```
from gensim.models import word2vec
wv = word2vec.Word2Vec.load_word2vec_format(PATH_TO_WORD_VECTORS, binary=True)
```

Then initialise the WSD object:

```
wsd_model = sense2vec.WSD(sv, wv, window=5, method='sim', filter_ctx=3)
```
This settings have the following meaning: the WSD model will extract at most `window`*2 words around the target word from the sentence as context and it will use only three most discriminative context words for disambiguation. 

Now you can disambiguate a word "table" in the sentence "They bought a table and chairs for kitchen" using `dis_text` function. As input it takes a sentence with space separated tokens, a target word, and start/end indices of the target word in the given sentence.

```
>>> wsd_model.dis_text("They bought a table and chairs for kitchen", "table", 14, 19)
(u'table#1', [0.15628162913257754, 0.54676466664654355])
```
It outputs its guess of the correct sense, as well as scores it assigned to all knwon senses during disambiguation. As one can see, it guessed the correct sense of the word "table" related to the furniture. The vector for this sense can be obtained from `sv["table#1"]`.

## Pretrained models
We provide several pretrained sense models accompanied by word and context vectors necessary for disambiguation. We have trained them on two corpora: English Wikipedia dump and UKWaC.

#### English Wikipedia

Word and context vectors:

```
wget /home/pelevina/experiment/model/public/wiki.words
wget /home/pelevina/experiment/model/public/wiki.contexts
```
Those vectors are of size 300, trained with CBOW model using 3-words context window, 3 iterations and minimum word frequency of 5.

Senses and probabilities induced using word2vec similarities between words:

```
wget /home/pelevina/experiment/model/public/wiki.senses.w2v
wget /home/pelevina/experiment/model/public/wiki.senses.w2v.probs
```

Senses and probabilities induced using JoBimText similarities between words:

```
wget /home/pelevina/experiment/model/public/wiki.senses.jbt
wget /home/pelevina/experiment/model/public/wiki.senses.jbt.probs
```

#### UKWaC

Word and context vectors:

```
wget /home/pelevina/experiment/model/public/ukwac.words
wget /home/pelevina/experiment/model/public/ukwac.contexts
```
Those vectors are of size 100, trained with CBOW model using 3-words context window, 3 iterations and minimum word frequency of 5.

Senses and probabilities induced using word2vec similarities between words:

```
wget /home/pelevina/experiment/model/public/ukwac.senses.w2v
wget /home/pelevina/experiment/model/public/ukwac.senses.w2v.probs
```

Senses and probabilities induced using JoBimText similarities between words:

```
wget /home/pelevina/experiment/model/public/ukwac.senses.jbt
wget /home/pelevina/experiment/model/public/ukwac.senses.jbt.probs
```

## Training a model

