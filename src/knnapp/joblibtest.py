from gensim import corpora, models,similarities
import logging as log
from joblib import Parallel,delayed,load,dump
import multiprocessing
import tempfile
import numpy as np
import os

def knn(i,filename):
	large_memmap = load(filename, mmap_mode='r+')	
	word = large_memmap[0].index2word[i]
	with open(str(i) + "_test.csv","w+") as f:
		f.write("test " + word)
if __name__ == "__main__":

	path = '/home/kurse/jm18magi/sensegram/resrc/GoogleNews-vectors-negative300.bin'

	log.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=log.INFO)

	model = models.Word2Vec.load_word2vec_format(path, binary=True,encoding='utf8')
	model_array = np.array([model])
	temp_folder = tempfile.mkdtemp(dir=os.getcwd())
	filename = os.path.join(temp_folder, 'joblib_test.mmap')
	if os.path.exists(filename): 
		os.unlink(filename)
	dump(model_array, filename)	
	Parallel(n_jobs=100)(delayed(knn)(i,filename)for i in range(100))
