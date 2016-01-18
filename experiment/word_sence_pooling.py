"""
Reads clusters (output of chinese-whispers) and creates a vector for each cluster 
(currently by averaging vectors of words in the cluster).
Result - a model of word senses
Saved in word2vec binary format
"""
# sense format: run#0 run#1 OK?

# Currently creates vectors for clusters in input.
# What if: 1. A word from original model doesn't have a cluster left after postprocessing 
# (all cluster were too small)
# 2. A word has only one cluster. Use the average of cluster (as currently) or the original word vector?

# saving with from sklearn.externals import joblib ?

# TODO: add a function to word2vec.py, that returns all senses of a word
# while word#i in model: result.append(model[word#i]), i+=1
# TODO: modify word2vec.py save function. It should iterate over index2word array
# instead of vocab set to preserve initial ordering

import codecs, pbar, time
from gensim.models import word2vec
from collections import defaultdict
import numpy as np
from pandas import read_csv
from operator import methodcaller

start = time.time()
CHUNK_LINES = 500000

# postprocessed clusters, file has header "word\tcid\tcluster\tisas"
clusters_path = 'dt/clusters-minsize5.csv' 

# Initialize Word2Vec object for sense vectors 
vocab_size = 107111 # number of clusters after postprocessing, read from postprocess stat output of chinese-whispers
# TODO: set up automated reading of vocab_size from postprocess.py output
vector_size = 200 # predefined
default_count = 10 # arbitrary, should be larger than min_count of vec object, which is 5 by default
senvec = word2vec.Word2Vec(size=vector_size, sorted_vocab=0)
senvec.syn0 = np.zeros((vocab_size, vector_size), dtype=np.float32) # numpy matrix initialization 

# load original model
model_path = 'model/text8_vectors.bin'
model_type = 'word2vec' # trained with word2vec or gensim
model_binary = True # True for binary, False for text (rare), only for word2vec

print("Loading original word model...")
if model_type == 'word2vec':
	wordvec = word2vec.Word2Vec.load_word2vec_format(model_path, binary=model_binary)
if model_type == 'gensim':
	wordvec = word2vec.Word2Vec.load(model_path)

print("Pooling cluster vectors...")
# open csv file with clusters
# file header is word\tcid\tcluster\tisas
reader = read_csv(clusters_path, encoding="utf-8", delimiter="\t", error_bad_lines=False, iterator=True, chunksize=CHUNK_LINES, doublequote=False, quotechar=u"\u0000")

sen_count_per_word = defaultdict(int)   
step = pbar.start_progressbar(vocab_size, 100)
i = 0  
for chunk in reader:
	for j, row in chunk.iterrows():
		# enumerate word senses from 0
		try:
			sen_word = str(row.word) + '#' + str(sen_count_per_word[row.word])
			sen_count_per_word[row.word] += 1
		except TypeError as e:
			z = e
			print row.word, str(sen_count_per_word[row.word]), z
			print row
			print i, row.cid, row.cluster
			continue
		
		cluster_words = [word for word, sim in map(methodcaller('split', ':'), row.cluster.split(","))]
		cluster_vectors = np.array([wordvec[word] for word in cluster_words])
		sen_vector = np.mean(cluster_vectors, axis=0)
		
		# copied from word2vec, modified
		def add_word(word, weights):
			word_id = len(senvec.vocab)
			if word in senvec.vocab:
				print("duplicate word sense '%s' in %s, ignoring all but first" % (word, clusters_path))
				return
			
			senvec.vocab[word] = word2vec.Vocab(index=word_id, count=default_count)
			senvec.syn0[word_id] = weights
			senvec.index2word.append(word)
			assert word == senvec.index2word[senvec.vocab[word].index]
		
		add_word(sen_word, sen_vector)
		if i%step==0:
			pbar.update_progressbar(i, vocab_size)
		i+=1
pbar.finish_progressbar()

if senvec.syn0.shape[0] != len(senvec.vocab):
	print("duplicate word senses detected, shrinking matrix size from %i to %i" % (senvec.syn0.shape[0], len(senvec.vocab)))
	senvec.syn0 = np.ascontiguousarray(senvec.syn0[:len(senvec.vocab)])
assert (len(senvec.vocab), senvec.vector_size) == senvec.syn0.shape

print("Calculated %sx%s matrix of sense vectors" % (senvec.syn0.shape))
end = time.time()
print("Calculation time: " + str(end - start))

print("Saving senvec model")
start = time.time()
senvec.save_word2vec_format(fname = 'model/text8_sense_vectors.bin', binary=True)
end = time.time()
print("Saving time: " + str(end - start))	


# TODO: test correct sense numbering 
# map(lambda x: x.split(), a) just useful  	
	
	
	