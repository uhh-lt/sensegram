# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
import numpy as np
cimport numpy as np
cimport cython.parallel
cimport cython
cimport openmp as omp
from gensim.models import Word2Vec
import logging as logger
import time


cdef public api double __pyx_v_x = float(0.0)
cpdef void compute_knn(float [:,:] vectors,float [:] result, int x = 100,int y  = 100, int knn=200,str schedule='static',int num_threads=4):
	cdef omp.omp_lock_t  lock_0
	cdef omp.omp_lock_t  lock_1
	cdef omp.omp_lock_t  lock_2
	cdef omp.omp_lock_t  lock_3
	omp.omp_init_lock(&lock_0)
	omp.omp_init_lock(&lock_1)
	omp.omp_init_lock(&lock_2)
	omp.omp_init_lock(&lock_3)
	cdef int i,j, id
	cdef int vectors_x = vectors.shape[0]
	cdef int vectors_y = vectors.shape[1]
	
	logger.info("max threads: " + str(omp.omp_get_max_threads()))
	#with nogil,cython.parallel.parallel(num_threads=10):
	#with nogil,cython.boundscheck(False),cython.wraparound(False):
		
	for i in range(0,x):
		#for i in cython.parallel.prange(x,schedule='static',num_threads=4):
		#with nogil,cython.parallel.parallel(num_threads=10):
		for j in range(0,y):
			with nogil,cython.parallel.parallel(num_threads=4):
				if(i != j):
					store_dot_product(a=vectors[i],b=vectors[j],result=result,result_index=i,lock=&lock_3)
	
				#	if( i % 4 == 0):
				#		store_dot_product(a=vectors[i],b=vectors[j],result=result,result_index=i,lock=&lock_0)
				#	elif(i % 4 == 1):
				#		store_dot_product(a=vectors[i],b=vectors[j],result=result,result_index=i,lock=&lock_1)
					#elif(i % 4 == 2):
					#	store_dot_product(a=vectors[i],b=vectors[j],result=result,result_index=i,lock=&lock_2)
					#elif(i % 4 == 3):
					#	store_dot_product(a=vectors[i],b=vectors[j],result=result,result_index=i,lock=&lock_3)
	omp.omp_destroy_lock(&lock_0)
	omp.omp_destroy_lock(&lock_1)
	omp.omp_destroy_lock(&lock_2)
	omp.omp_destroy_lock(&lock_3)
	




cdef void store_dot_product(float [:] a, float [:] b, float [:] result, int result_index, omp.omp_lock_t * lock) nogil:	
#	if(a.shape[0] != b.shape[0]):
#		return
	cdef float tmp = 0.0
	cdef int i,x
	x = a.shape[0]
	for i in range(0,x):
		tmp = tmp + a[i]*b[i]
	#omp.omp_set_lock(lock)
	result[result_index] += tmp
	#omp.omp_unset_lock(lock)
	
"""
path = '/home/kurse/jm18magi/sensegram/resrc/GoogleNews-vectors-negative300.bin'
logger.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logger.INFO)

model = Word2Vec.load_word2vec_format(path, binary=True,encoding='utf8')
result = np.zeros([3000000],dtype=np.float32)

compute_knn(vectors = model.syn0,result = result,x=1000,y=10000)

for i in range(0,10):
	logger.info(result[i])

"""	
