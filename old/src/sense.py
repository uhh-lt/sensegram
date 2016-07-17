#!/usr/bin/env python
#
# Author: Jonas Molina Ramirez, Kai Steinert
# Version: 0.1
# Date: 12/11/2015
#
import time
from gensim import corpora, models, similarities
import logging
import gensim as gs
import multiprocessing as mp

path = '../resrc/GoogleNews-vectors-negative300.bin'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def logInfo(start, end,txt):
	logging.info(txt + str(end-start) + ' seconds')

def load_model():
    #load model from disk
    start = time.time()
    model = models.Word2Vec.load_word2vec_format(path, binary=True,encoding='utf8')
    end = time.time()
    logInfo(start,end,'Loaded model in ')
    return model


def compute_knn(model, start_idx, end_idx, part, nn):
    #compute thesaurus of 200 nearest neighbours
    start = time.time()
    thesaurus = dict((k,dict(model.most_similar(k,topn=nn)))
                     for k in model.index2word[start_idx:end_idx])
    file = open('../resrc/similarities_' + str(part) + '.csv', 'w')
    [file.write(k.encode('utf-8') + '\t' + w.encode('utf-8') + '\t' + str(s) + '\n') for k in thesaurus for (w, s) in thesaurus[k].items()]
    file.close()
    end = time.time()
    logInfo(start,end,'Computed thesaurus in ')

    # counter = 0
    # knn = []
    # file = open('../resrc/similarities_' + part + '.csv', 'w')
    # for k in model.index2word[start:end]:
    #     knn = knn + [(k, dict(model.most_similar(k,topn=200)))] 
    #     if (counter == chunksize):
    #         thesaurus = dict(knn)
    #         [file.write(k.encode('utf-8') + '\t' + w.encode('utf-8') + '\t' + str(s) + '\n') for k in thesaurus for (w, s) in thesaurus[k].items()]
    #         knn = []
    #         counter = 0
    #     else:
    #         counter = counter +1
    # else:
    #     thesaurus = dict(knn)
    #     [file.write(k.encode('utf-8') + '\t' + w.encode('utf-8') + '\t' + str(s) + '\n') for k in thesaurus for (w, s) in thesaurus[k].items()]
    # file.close()
    #thesaurus = dict((k,dict(model.most_similar(k,topn=200))) for k in model.index2word)


def start_serial():
    model = load_model()
    modelsize = len(model.index2word)
    chunksize = 10000
    chunks = range(0, modelsize, chunksize)
    chunks.append(modelsize)
    [compute_knn(model, chunks[x-1], chunks[x], x, 200) for x in range(1, len(chunks))]

# UNTESTED !
def start_parallel():
    model = load_model()
    modelsize = len(model.index2word)
    chunksize = 10000
    chunks = range(0, modelsize, chunksize)
    chunks.append(modelsize)

    pool = mp.Pool(processes=16)
    [pool.apply_async(compute_knn, args=(model, chunks[x-1], chunks[x], x, 200)) for x in range(1, len(chunks))]
