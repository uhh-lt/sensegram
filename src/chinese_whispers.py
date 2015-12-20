#!/usr/bin/env python
#
# Author: Jonas Molina Ramirez, Kai Steinert
# Version: 0.1
# Date: 12/11/2015
#

import re
from itertools import groupby
from operator import itemgetter

def comp_rank(class_labels, thesaurus):
    for i in range(0, len(class_labels)):
        c = class_labels[i][0]
        classes = dict() # potential classes for one node
        for nn, sim in thesaurus[c].items():
            if(nn in classes):
                classes[nn] = classes[nn] + sim
            else:
                classes[nn] = sim

        class_labels[i] = (class_labels[i][0], max(classes.items(), key=itemgetter(1))[0])
    print class_labels


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

    comp_rank(class_labels, thesaurus)

