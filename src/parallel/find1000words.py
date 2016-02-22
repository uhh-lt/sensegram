from gensim import models
import codecs
#path = '/home/kurse/jm18magi/sens=egram/resrc/GoogleNews-vectors-negative300.bin'
#model = models.Word2Vec.load_word2vec_format(path, binary=True,encoding='utf8')
def writewords(model):
	topWords = codecs.open('top1000en.txt','r')
	with codecs.open('top1000en_vectors.txt','w','utf-8') as vector_file:
		for word in topWords:
			tmp = word.replace('\n','')
			index = model.vocab[tmp].index
			vector_file(tmp + ' ' + model.syn0 + '\n')
