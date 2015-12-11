#!/usr/bin/env python
#
# Author: Jonas Molina Ramirez, Kai Steinert
# Version: 0.1
# Date: 12/11/2015
#
from gensim import corpora, models, similarities
import logging
import gensim
#import pandas as pd
path = 'GoogleNews-vectors-negative300.bin'
#path2 = '../english.rcv1.100.embeddings.gz'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#load model from disk
start = time.time()
model = models.Word2Vec.load_word2vec_format(path, binary=True,encoding='utf8')
end = time.time()

def logInfo(start, end,txt):
	logging.info(txt + str(end-start) + ' seconds')

logInfo(strat,end,'Loaded model in ')
#compute thesaurus of 200 nearest neighbours 
start = time.time()
thesaurus = dict((k,model.most_similar(k,topn=200)) for k in model.index2word[:5])
end = time.time()
logInfo(start,end,'Computed thesaurus in ')

