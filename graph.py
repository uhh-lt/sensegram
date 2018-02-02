from bidict import bidict
import numpy as np
from time import time
from traceback import format_exc
import codecs
from scipy.sparse import csr_matrix
from math import floor

WEIGHT_COEF = 10000.

class CRSGraph(object):
    """ A graph based on the CSR sparse matrix data structure. """

    def __init__(self, neighbors_fpath):
        self._graph, self.index = self._load(neighbors_fpath) 
      
    
    def _get_or_add(self, dictionary, value):
        """ Gets the key associated with the value if exists. 
        Otherwiese inserts the value eq. to the length of the 
        dictionary and returns the key. """

        if value not in dictionary:
            value_idx = len(dictionary)
            dictionary[value] = len(dictionary)
        else:
            value_idx = dictionary[value]

        return value_idx

    
    def _load(self, neighbors_fpath):   
        tic = time()
        with codecs.open(neighbors_fpath, "r", "utf-8") as graph:
            src_lst = []
            dst_lst = []
            data_lst = []
            index = bidict()
            word_dict = {}
            for i, line in enumerate(graph):                
                if i % 10000000 == 0 and i != 0: print(i)
                try:
                    src, dst, weight = line.split("\t")
                    src = src.strip()
                    dst = dst.strip()
                    src_idx = self._get_or_add(index, src)
                    dst_idx = self._get_or_add(index, dst) 

                    src_lst.append(int(src_idx))
                    dst_lst.append(int(dst_idx))
                    data_lst.append(np.int16(floor(float(weight) * WEIGHT_COEF)))
                except:
                    print(format_exc())
                    print("Bad line:", line)

        rows = np.array(src_lst)
        cols = np.array(dst_lst)
        data = np.array(data_lst, dtype=np.int16)
        graph = csr_matrix( (data, (rows, cols)), shape=(len(index),len(index)), dtype=np.int16 )       
        print("Loaded in {:f} sec.".format(time() - tic))

        return graph, index 

    def get_neighbors(self, word):
        """ Returns a dictionary with nearest neighbors. """

        idx_i = self.index[word]
        data_i = self._graph[idx_i].data
        nns = {self.index.inv[idx_j]: data_i[j] 
               for j, idx_j in enumerate(self._graph[idx_i].indices)}
        
        return nns
       
    def get_weight(self, word_i, word_j):
        """ Returns weight of a pair of elements. """

        idx_i = self.index[word_i]
        idx_j = self.index[word_j]
        slice_i = self._graph[idx_i]
        r = np.where(slice_i.indices == idx_j)[0]
        if r.size > 0:
            return slice_i.data[r[0]]
        else:
            return 0.0


