# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
import numpy as np
cimport numpy as np
cimport cython.parallel
cimport cython
from cython.parallel cimport prange
cimport openmp as omp
from gensim.models import Word2Vec
import logging as logger
import time


cdef public api double __pyx_v_x = float(0.0)

cpdef void compute_knn(float [:,:] vectors,float [:] result, int x = 100,int y  = 100, int knn=200):
	cdef int i,j, id
	cdef int vectors_x = vectors.shape[0]
	cdef int vectors_y = vectors.shape[1]
	
	logger.info("max threads: " + str(omp.omp_get_max_threads()))
	#with nogil,cython.parallel.parallel(num_threads=10):
	#with nogil,cython.boundscheck(False),cython.wraparound(False):
#	with nogil:	
	for i in range(0,x):
			iter_neighbours(vectors=vectors,result=result,i=i,no_of_vectors=vectors_x)

cpdef void test(float[:,:] vectors,int i, int no_of_vectors):
	cdef np.ndarray result = np.zeros([3000000],dtype=np.float32)
	iter_neighbours(vectors,result,i,no_of_vectors)
	

cpdef void iter_neighbours(float[:,:] vectors, float[:] result, int i,int no_of_vectors):
	if( no_of_vectors <=i):
		return
	cdef int index, j,length
	cdef np.float32_t tmp
	length = vectors.shape[1]
	#with nogil,cython.parallel.parallel(num_threads=4):
	# compute only a triangular matrix
	for j in range(i,no_of_vectors):
		tmp = dot_product(a=vectors[i],b=vectors[j],length=length)
		tmp = tmp / (norm(vectors[i],length) * norm(vectors[j],length))
		result[j] = result[j] + tmp

	return
	
#Parameter:
# result_index: cell index of result where the result should be stored
#
#
cpdef float dot_product(float [:] a, float [:] b,int length) nogil:	
	cdef float tmp = 0.0
	cdef int i
	for i in range(0,length):
		tmp = tmp + a[i]*b[i]
	#omp.omp_set_lock(lock)
	return tmp
	#omp.omp_unset_lock(lock)

import math
cdef extern from "math.h":
	double sqrt(double m)

cpdef double norm(float[:] a,int length):
	cdef tmp = 0.0
	cdef int i
	for i in range(0,length):
		tmp = tmp + a[i]*a[i]
	return sqrt(tmp)

		
"""
path = '/home/kurse/jm18magi/sensegram/resrc/GoogleNews-vectors-negative300.bin'
logger.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logger.INFO)

model = Word2Vec.load_word2vec_format(path, binary=True,encoding='utf8')
result = np.zeros([3000000],dtype=np.float32)

compute_knn(vectors = model.syn0,result = result,x=1000,y=10000)

for i in range(0,10):
	logger.info(result[i])

"""	
