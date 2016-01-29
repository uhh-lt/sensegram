from gensim.models import word2vec
from operator import methodcaller
import operator
import numpy as np

def get_senses(smodel, word):
	""" returns all available senses for a given word.
	example: 'mouse' -> ['mouse#0', 'mouse#1', 'mouse#2']
	Assumption: senses use continuous numbering"""
	senses = []
	for i in range(0,200):
		try:
			smodel[word+'#'+str(i)]
			senses.append(word+'#'+str(i))
		except KeyError:
			break
	return senses
	
# def get_senses2(smodel, word):
# 	""" Alternative implementation of get_senses
# 	No such assumption as in get_senses(), but 1000 times slower"""
# 	senses = []
# 	pat = re.compile("^"+word+"#\d+$")
# 	for w, obj in smodel.vocab.items():
# 		if re.match(pat, w):
# 			senses.append(w)
# 	return senses
	
# useful: timer = timeit.Timer("get_senses(m,'python')", "from __main__ import get_senses, m")
# timer.timeit(100)

class WSD(object):
	def __init__(self, path_to_sense_model, path_to_context_model):
		self.vs = word2vec.Word2Vec.load_word2vec_format(path_to_sense_model, binary=True)
		self.vc = word2vec.Word2Vec.load_word2vec_format(path_to_context_model, binary=True)
		
	def get_context(self, text, target_position, window):
		""" returns a list of words surrounding the target positioned at [start:end] in the text 
		target_pos is a string 'start,end' 
		window=5 means 5 words on the left + 5 words on the right are returned, if they exist"""
		start, end = [int(x) for x in target_position.split(',')]
		l, r = text[:start].split(), text[end:].split()
	
		# words in our model contain only 26 english lowercase letters
		l, r = filter(methodcaller('isalpha'), l), filter(methodcaller('isalpha'), r)
		return [w.lower() for w in l[-window:] + r[:window]]
		
		# TODO: in our model train set all numbers are spelled out (2 = two). Take it into account when predicting.
		# TODO: clear test example from stop words?
	
	# NOTE: if model is loaded with norm_only=True (that's default), then
	# both syn0 and syn0norm contain normalized vectors. In this case model['word'] shortcut
	# also returns a normalized vector
	def __logprob__(self, ctx, vsense):
		""" returns P(vsense|ctx), where vsense is a vector, ctx is a word """
		#vctx = vc.syn0norm[vc.vocab[ctx].index]
		vctx = self.vc[ctx]
		return 1.0 / (1.0 + np.exp(-np.dot(vctx,vsense)))
		
	def __prob__(self, ctx, vsense):
		""" returns probability of a sense (vector) in a given context (list of words)"""
		return reduce(operator.mul, [self.__logprob__(c, vsense) for c in ctx if c in self.vc], 1)
		
	def dis_text(self, text, pos, word):
		""" disambiguates the sense of a word in given text
			text - a tokenized string ("Obviously , it was cold .")
			pos - position of a word in text ("12,14")
			word  - word to be disambiguated ("it")
			returns None if word is not covered by the model"""
		ctx = self.get_context(text, pos, 10)
		return self.dis_context(ctx, word)

	def dis_context(self, context, word):
		""" disambiguates the sense of a word in given context
			context - a list of context words
			word  - word to be disambiguated
			returns None if word is not covered by the model"""
		senses = get_senses(self.vs, word)
		if len(senses)==0:
			return None
		probs = [self.__prob__(context, self.vs[sense]) for sense in senses]
		return senses[np.argmax(probs)], max(probs)
		# TODO: manage situation where senses is empty
# Example:
# text = "However , the term mouse can also be applied to species outside of this genus . Mouse often refers to any small muroid rodent , while rat refers to larger muroid rodents"
# pos = "80,85"
# word = "mouse"

		
	