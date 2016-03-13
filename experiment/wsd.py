from operator import methodcaller, mul
import math
import numpy as np
from gensim.models import word2vec

def get_senses(smodel, word):
    """ returns all available senses for a given word.
    example: 'mouse' -> ['mouse#0', 'mouse#1', 'mouse#2']
    Assumption: senses use continuous numbering"""
    senses = []
    for i in range(0,200):
        try:
            smodel[word + u'#' + unicode(i)]
            senses.append(word + u'#' + unicode(i))
        except KeyError:
            break
    return senses
    
# def get_senses2(smodel, word):
#   """ Alternative implementation of get_senses
#   No such assumption as in get_senses(), but 1000 times slower"""
#   senses = []
#   pat = re.compile("^"+word+"#\d+$")
#   for w, obj in smodel.vocab.items():
#       if re.match(pat, w):
#           senses.append(w)
#   return senses

class WSD(object):
    def __init__(self, path_to_sense_model, path_to_context_model, window=10):
        self.vs = word2vec.Word2Vec.load_word2vec_format(path_to_sense_model, binary=True)
        self.vc = word2vec.Word2Vec.load_word2vec_format(path_to_context_model, binary=True)
        self.window = window
        
    def get_context(self, text, target_position):
        """ returns a list of words surrounding the target positioned at [start:end] in the text 
        target_pos is a string 'start,end' 
        window=5 means 5 words on the left + 5 words on the right are returned, if they exist"""
        start, end = [int(x) for x in target_position.split(',')]
        l, r = text[:start].split(), text[end:].split()
    
        # it only makes sense to use context for which we have vectors
        l = [ctx for ctx in l if ctx in self.vc.vocab]
        r = [ctx for ctx in r if ctx in self.vc.vocab]
        return l[-self.window:] + r[:self.window]
        
        # TODO: clear test example from stop words?
    
    # NOTE: if model is loaded with norm_only=True (that's default), then
    # both syn0 and syn0norm contain normalized vectors. 
    # In this case model['word'] shortcut also returns a normalized vector
    def __logprob__(self, ctx, vsense):
        """ returns P(vsense|ctx), where vsense is a vector, ctx is a word """
        #vctx = vc.syn0norm[vc.vocab[ctx].index]
        vctx = self.vc[ctx]
        return 1.0 / (1.0 + np.exp(-np.dot(vctx,vsense)))
        
    def __prob__(self, ctx, vsense):
        """ returns probability of a sense (vector) in a given context (list of words)"""
        return reduce(mul, [self.__logprob__(c, vsense) for c in ctx if c in self.vc], 1)

    def norm(self, a):
        s = sum(a)
        return [float(p)/s for p in a]

    def entropy(self, dist):
        # give some probability to zero elements
        m = float(max(dist))
        dist = [m/1000 if p == 0 else p for p in dist]
        s = len(dist)
        return -sum([p*math.log(p,2) for p in self.norm(dist)])/s

    def diff_confidence(self, dist):
        """ measure of 'flatness' of a distribution. ~ sum of differences between max prob and others.
        The higher the better."""
        m = max(dist)
        dist = [float(x)/m for x in dist]
        dist = [1 - x for x in dist]
        return sum(dist)/len(dist)
        
    def dis_text(self, text, pos, word):
        """ disambiguates the sense of a word in given text
            text - a tokenized string ("Obviously , it was cold .")
            pos - position of a word in text ("12,14")
            word  - word to be disambiguated ("it")
            returns None if word is not covered by the model"""
        ctx = self.get_context(text, pos)
        return self.dis_context(ctx, word)

    def dis_context(self, context, word):
        """ disambiguates the sense of a word in given context
            context - a list of context words
            word  - word to be disambiguated
            returns None if word is not covered by the model"""
        senses = get_senses(self.vs, word)
        if len(senses)==0: # means we don't know any sense for this word
            return None 
        context = [ctx for ctx in context if ctx in self.vc] # this check happens in __prob__
        prob_dist = [self.__prob__(context, self.vs[sense]) for sense in senses]
        e_confidence = self.entropy(prob_dist)
        diff_confidence = self.diff_confidence(prob_dist)
        # return sense (word#id), probability, entropy confidence, differences confidence, prob_dist and length of context
        return senses[np.argmax(prob_dist)], prob_dist, e_confidence, diff_confidence, len(context)
# Example:
# text = "However , the term mouse can also be applied to species outside of this genus . Mouse often refers to any small muroid rodent , while rat refers to larger muroid rodents"
# pos = "80,85"
# word = "mouse"

# dis_context follows approach: window first, filter after
# dis_text follows approach: filter first, window later
# TODO: find consistent approach!

        
    