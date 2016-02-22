from gensim import models
import logging as logger
logger.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logger.INFO)

def load_model():
	path = '/home/kurse/jm18magi/sensegram/resrc/GoogleNews-vectors-negative300.bin'
	model = models.Word2Vec.load_word2vec_format(path, binary=True,encoding='utf8')
	return model
