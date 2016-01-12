#!/usr/bin/env python
#
# Author: Jonas Molina Ramirez, Kai Steinert
# Version: 0.1
# Date: 12/11/2015
#
import time
from datetime import datetime
from gensim import corpora, models, similarities
import logging
import gensim as gs
import multiprocessing as mp
import os

path = '../resrc/GoogleNews-vectors-negative300.bin'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def run_parallel(knn,model,numberOfFiles,end_idx,start_idx=0):
	# create a list of locks. each lock is reserved for a file
	# and identified by the dictionary's key 
	locks = dict([(id,mp.Lock()) for id in range(0,numberOfFiles)])

	# auxiliary counter for assigning locks to processes
	counter = 0

	# counter the number of instantiated processes
	processCounter = 0
	start = time.time()
	logging.info("Start computation of KNN: \n" + str(datetime.fromtimestamp(start)))
	processes = []
	for word in model.index2word[start_idx:end_idx]:
		#parallel(model,item,counter,5,locks.get(counter))
		p = mp.Process(target=parallel,args=(model,word,counter,knn,locks.get(counter)))
		p.start()
		processes.append(p)
		counter += 1
		processCounter += 1
		if(counter%numberOfFiles==0):
			#processCounter = processCounter + counter
			counter = 0
	# wait for processes to finish
	for proc in processes:
		proc.join()
	end = time.time()
	overallSize = 0
	for dirpath,dirnames,filenames in os.walk("tmp"):
		for f in filenames:
			#print f + '\n'
			fp = os.path.join(dirpath,f)
			overallSize += os.path.getsize(fp)				
	
	logging.info("End computation of KNN: \n" + str(datetime.fromtimestamp(end)))
	logging.info("Constructed " + str(processCounter) + " processes.")
	logInfo(start,end,'Wrote files in ')
	logging.info("Overall file size: " + str(overallSize/1024) + "kb")


def logInfo(start, end,txt):
	logging.info(txt + str(end-start) + ' seconds')

def load_model():
	start = time.time()
	model = models.Word2Vec.load_word2vec_format(path, binary=True,encoding='utf8')
	end = time.time()
	logInfo(start,end,'Loaded model in ')
	return model

def parallel(model,word,id,nn,lock):
	pid = str(os.getpid())
	logging.info("Start Pid: " + pid)
	neighbours = dict(model.most_similar(word,topn=nn))
	text = [('\t' + w.encode('utf-8') + '\t' + str(s) + '\n') for (w,s) in neighbours.iteritems()]
	lock.acquire()
	with open("tmp/test_" + str(id) + ".csv", "a") as myfile:
		for line in text:
			myfile.write(word.encode('utf-8') + line)
	lock.release()
	logging.info("End Pid: " + pid)



