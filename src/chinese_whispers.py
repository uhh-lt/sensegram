#!/usr/bin/env python
#
# Author: Jonas Molina Ramirez, Kai Steinert
# Version: 0.1
# Date: 12/11/2015
#

<<<<<<< HEAD
import re,pprint as pp
=======
import random
import re
>>>>>>> 66cea49e349219dd41c98ebad41cf91af6a3b52b
from itertools import groupby
from operator import itemgetter

# def comp_rank(class_labels, thesaurus):
#     for i in range(0, len(class_labels)):
#         c = class_labels[i][0]
#         classes = dict() # potential classes for one node
#         for nn, sim in thesaurus[c].items():
#             if(nn in classes):
#                 classes[nn] = classes[nn] + sim
#             else:
#                 classes[nn] = sim

#         class_labels[i] = (class_labels[i][0], max(classes.items(), key=itemgetter(1))[0])
#     print class_labels
# [(nodes_classes[word], sim) for (word, sim) in thesaurus[node].items()]
def comp_rank(thesaurus):
    nodes_classes = dict((w, w) for w in thesaurus)
    for iterations in range(0, 10):
        random.shuffle(nodes_classes.items())
        potential_classes = [groupby(sorted([(nodes_classes[word], sim)
                                              for (word, sim) in thesaurus[node].items()]), key=itemgetter(0))
                              for (node, node_class) in nodes_classes.items()]
        [[(word, lst) in node] for node in potential_classes]

[[(nodes_classes[word], sim) for (word, sim) in thesaurus[node].items()] for (node, node_class) in nodes_classes.items()]

#[(k, list(v))for (k, v) in groupby(sorted(test2), key=itemgetter(0))]
#    test_data = {'wer':{'wie':1, 'was':2, 'wann':3, 'wo':4}, 'wie':{'wer':1, 'was':5, 'wann':6, 'wo':7}, 'was':{'wer':2, 'wie':5, 'wann':8, 'wo':9}, 'wann':{'wer':3, 'wie':6, 'was':8, 'wo':10}, 'wo':{'wer':4, 'wie':7, 'was':9, 'wann':10}}

#'wer':, 'wie':, 'was':, 'wann':, 'wo':

def chinese_whispers():

    fname = "../resrc/similarities.csv"
    with open(fname) as f:
        raw_similarities = f.readlines()

    tab_pattern = re.compile('\t')

    similarities = [tab_pattern.split(entry.replace('\n', '')) for entry in raw_similarities]

    knn=200

    thesaurus = dict((similarities[i*knn][0], dict((k, v) for (d, k, v) in similarities[i*knn:i*knn+knn]))
                     for i in range(0 ,len(similarities)/knn))

    class_labels = [(w, w) for w in thesaurus]

    comp_rank(thesaurus)


def filter_clusters(fname,min_cluster_size=0):
	
	with open(fname) as f:
		raw_clusters = f.readlines()
	
	tab_pattern = re.compile('\t')
	
	clusters = [tab_pattern.split(entry.replace('\n','')) for entry in raw_clusters]
	filtered_clusters = []
	for k in clusters:
		# split cluster into its elements
		split = k[3].split('  ')
		if(len(split)>min_cluster_size):
						
			filtered_clusters = filtered_clusters + [(k[2],[re.split(':0\.\d+',x)[0] for x in split])]

	return filtered_clusters
