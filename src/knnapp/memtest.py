
from gensim import corpora, models, similarities
import multiprocessing as mp
from multiprocessing.managers import BaseManager,NamespaceProxy
import logging as log
import sys
import threading
import time
log.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=log.DEBUG)

path = '/home/kurse/jm18magi/sensegram/resrc/GoogleNews-vectors-negative300.bin'

model = models.Word2Vec.load_word2vec_format(path, binary=True,encoding='utf8')

class MyThread(threading.Thread):
	def __init__(self,word,dir):
		threading.Thread.__init__(self)
		self.word = word
		self.dir = dir
	def run(self):
		with open( self.dir,"w+") as f:
			neighbours = model.most_similar(self.word,topn=200)
			for (w,s) in neighbours:
				f.write(self.word + '\t' + w + '\t' + str(s) + '\n')
		
def f(start,end,model):
	init_threads(start,end,model)

def get_word(index):
	return model.index2word[index]

def init_threads(start_idx,end_idx,model):
	my_threads =[]
	startTime = time.time()
	
	for i in range(start_idx,end_idx):
		word =  model.index2word[i]
		t = MyThread(word,os.getcwd() + "/" +str(i) + "_test.csv")
		my_threads.append(t)
		t.start()
	for t in my_threads:
		t.join()
	endTime = time.time()
	logInfo(startTime,endTime,"Time to start and join threads: ")
# Logs the difference between end and start 
def logInfo(start, end,txt):
	log.info(txt + str(end-start) + ' seconds')

def sequential(word,index):
	neighbours = model.most_similar(word,topn=200)
#	log.debug("word: " + word + ", len neighbours: " + str(len(neighbours)))

	with open("seq/"+str(index) +"_test.csv","w+")as myfile:
#		log.debug("#neighbours for " + str(index) + "_test.csv: ")
		for (w,s) in neighbours:
			myfile.write(word + '\t' + w + '\t' + str(s) + '\n')
	del neighbours
	return word
def sequential_test(end_idx):
	startTime = time.time()
	for i in range(end_idx): 
		word = model.index2word[i]
		sequential(word,i)
	endTime = time.time()
	logInfo(startTime,endTime,"Time for sequential computation: ")
		

if __name__== "__main__":
	log.debug("model size in bytes: " + str(sys.getsizeof(model)))
	
	#sequential_test(1000)

#	cpus = mp.cpu_count()
#	log.info("CPU count: " + str(cpus))
	pool = mp.Pool(processes=mp.cpu_count()*100)
	
	pool_index = 10
	word_index = 100
	for i in range(pool_index):
		start = i*(word_index/pool_index)
		end = start + (word_index/pool_index)
		log.debug("start: " + str(start) + ", end: " + str(end))
		pool.apply_async(init_threads,(start,end,model,),callback=cb)
	pool.close()
#	max_children = -1
	pool.join()
