from bidict import bidict
import numpy as np
from time import time
from traceback import format_exc
import codecs
from scipy.sparse import csr_matrix
from math import floor


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
                    data_lst.append(np.int16(floor(float(weight) * 10000.)))
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
        idx_i = self.index[word]
        
#         nns = {self.index.inv[idx_j]: self._graph[idx_i].data[j] 
#             for j, idx_j in enumerate(self._graph[idx_i].indices)}
        data_i = self._graph[idx_i].data
        nns = {self.index.inv[idx_j]: data_i[j] 
               for j, idx_j in enumerate(self._graph[idx_i].indices)}
        
        return nns
       
    def get_weight(self, word_i, word_j):
        return 0.0



from itertools import zip_longest
import argparse, sys, subprocess
from utils.common import exists
from os.path import basename
import gensim
import gzip
from gensim.models.keyedvectors import KeyedVectors
from gensim.utils import tokenize
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from time import time
import numpy as np
from chinese_whispers import chinese_whispers, aggregate_clusters
import codecs
import networkx as nx
from multiprocessing import Pool
from os.path import join
from collections import defaultdict
import codecs 
from time import time

import filter_clusters
import vector_representations.build_sense_vectors
from utils.common import ensure_dir
import pcz

verbose = True


def get_ego_network(ego):
    tic = time()
    ego_network = nx.Graph(name=ego)
    
    # Add related and substring nodes
    substring_nodes = []
    for j, node in enumerate(G.index):
        if ego.lower() == node.lower():
            nns_node = G.get_neighbors(node)
            ego_nodes = [(rn, {"weight": w}) for rn, w in nns_node.items()]
            ego_network.add_nodes_from(ego_nodes)
        else:
            if "_" not in node: continue
            if node.startswith(ego + "_") or node.endswith("_" + ego):
                substring_nodes.append( (node, {"weight": 9999}) )
    ego_network.add_nodes_from(substring_nodes)
    
    # Find edges of the ego network
    for r_node in ego_network:
        related_related_nodes = G.get_neighbors(r_node)
        related_related_nodes_ego = sorted(
            [(related_related_nodes[rr_node], rr_node) for rr_node in related_related_nodes if rr_node in ego_network],
            reverse=True)[:n]
        related_edges = [(r_node, rr_node, {"weight": w}) for w, rr_node in  related_related_nodes_ego]
        ego_network.add_edges_from(related_edges)
    
    chinese_whispers(ego_network, weighting="top", iterations=20)
    if verbose: print("{}\t{:f} sec.".format(ego, time()-tic))

    return ego_network


G = None
n = None


def ego_network_clustering(neighbors_fpath, clusters_fpath, max_related=300, num_cores=32):
    global G
    global n
    G = graph 
    
    with codecs.open(clusters_fpath, "w", "utf-8") as output, Pool(num_cores) as pool:
        output.write("word\tcid\tcluster\tisas\n")

        for i, ego_network in enumerate(pool.imap_unordered(get_ego_network, G.index)):
            if i % 100 == 0: print(i, "ego networks processed")
            sense_num = 1
            for label, cluster in sorted(aggregate_clusters(ego_network).items(), key=lambda e: len(e[1]), reverse=True):
                output.write("{}\t{}\t{}\t\n".format(
                    ego_network.name,
                    sense_num,
                    ", ".join( ["{}:{:.4f}".format(c_node, ego_network.node[c_node]["weight"]) for c_node in cluster] )
                ))
                sense_num += 1
    print("Clusters:", clusters_fpath)

       
neighbors_fpath = "model/wikipedia.txt.graph"
clusters_fpath = "model/tmp-wikipedia-2.txt"
graph = CRSGraph(neighbors_fpath)
ego_network_clustering(neighbors_fpath, clusters_fpath, max_related=300, num_cores=40)

