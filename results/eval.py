from gensim import models
import re
import model
import numpy as np
from scipy.spatial.distance import cosine

def compute_context(model):
	results = []
	with open('homonym_examples.txt','r',encoding='utf-8') as input:
		for line in input:
			content = re.split('\t',line)
			print(len(content))
			for i in range(1,len(content)):
				words = re.split(' ',content[i].replace('.','').replace(',','').replace('!','').replace('?',''))
				vector_sum = np.zeros(300)
				words = [w.replace('\n','') for w in words]
				for word in words:
					if(word != content[0]):
						try:
							vector_sum += model.syn0[model.vocab[word].index]
						except KeyError:
							print("Couldn't find key: " + word)
				results = results +  [(content[0],vector_sum,words)]
			#	print(content[0] + ": " + str(vector_sum))	
	return results

def evaluate(context_list):
	results = []
	for (word,vec,ctx) in context_list:
		with open('pool.csv','r',encoding='utf-8') as input:
	#		input.seek(0)
			min_dist = ('',-1,float('inf'))
			for line in input:
				content = re.split('\t',line.replace('\n',''))
				features = re.split(' ',content[3].replace(',','').replace('[','').replace(']',''))
				features = [float(f) for f in features if f != '']
				cos = cosine(vec,features)
				if( cos < min_dist[2] ):
					print(word + ' '  + str(cos) +'   ' + str(min_dist[2]))
					min_dist = (word,content[0],cos) # content[0] gives the index of the cluster in pool.csv
			results = results + [min_dist]
	return results	
if __name__ == '__main__':
	evaluate(model.load_model())
		
