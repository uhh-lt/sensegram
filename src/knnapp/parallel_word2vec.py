from gensim import models
import logging as log
from cython.parallel import parallel,prange
from libc.stdlib import abort,malloc,free
cimport openmp

path = '/home/kurse/jm18magi/sensegram/resrc/GoogleNews-vectors-negative300.bin'
log.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=log.INFO)

class ParallelWord2Vec(models.Word2Vec):
	pass


def compute_knn():
	cdef int num_threads
	openmp.omp_set_dynamic(1)
	with nogil,parallel():
		num_threads = openmp.omp_get_num_threads()
	return num_threads

if __name__ == "__main__":
	pwv = ParallelWord2Vec()
#	pwv = models.Word2Vec()
	#model = ParallelWord2Vec.load_word2vec_format(path, binary=True,encoding='utf8')
	#print(len(model.index2word))	
	print(str(compute_knn()))
