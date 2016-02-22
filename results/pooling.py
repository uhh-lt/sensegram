import sys
import re
from gensim import models
import logging as logger
logger.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logger.INFO)
import numpy as np
import string
def compute_average(file,model):
	with open('pool.csv','w',encoding='utf-8') as output:
		with open(file,'r',encoding='utf-8') as input:
			for line in input:
				content = re.split('\t',line)
				word_string  = content[len(content)-1]
				#print(str(word_string))
				word_string = word_string.replace('\n','') # remove newline character
				words = re.split(', ', word_string)
				words = words[:len(words)-1]#remove white space from list
				#print(words)
				sum = np.zeros(300) 
				counter = 0
				for word in words:
					index = model.vocab[word].index #find index of word 
					sum = sum +  model.syn0[index] #find feature vector and add to sum
					counter += 1
				if(counter == 0):
					print("Something wen wrong: no words processed for line: " + line)
					return
				average = sum / counter
				output.write(line.replace('\n','') + ' \t' + str(average).replace('\n','') + '\n')

if __name__ == "__main__":
	path = '/home/kurse/jm18magi/sensegram/resrc/GoogleNews-vectors-negative300.bin'
	
	length = len(sys.argv)
	if length == 2:
		model = models.Word2Vec.load_word2vec_format(path, binary=True,encoding='utf8')
		compute_average(sys.argv[1],model)
	else:
		print("Error: number of input arguments is " + str(length - 1))
