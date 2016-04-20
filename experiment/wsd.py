""" There are three WSD methods to handle context:
'sep'-- probability of a sense is computed for each single word in context. Then probabilities are multiplied.
'avg'-- First all context words are pulled into a single vector, then a probability of sense vector with given context vector is caculated
'sim'-- Same as avg, but calculates cosine similarity between sense vector and context vector.

Context filtering:
-filter_ctx option. Currently implemented only for sim method. Uses only (n) most informative words for disambiguation. "Informativeness" is measured as difference between most similar and least similar sense for a word. Large difference -> important word. Takes 2 most important words (best value as proven by evaluation)
"""

from operator import methodcaller, mul, itemgetter
import math
import numpy as np
from scipy.spatial.distance import cosine
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
    def __init__(self, path_to_sense_model, path_to_context_model, window=10, method="sep", filter_ctx=False):
        self.vs = word2vec.Word2Vec.load_word2vec_format(path_to_sense_model, binary=True)
        self.vc = word2vec.Word2Vec.load_word2vec_format(path_to_context_model, binary=True)
        self.window = window
        self.ctx_method = method
        self.filter_ctx = filter_ctx
        
        print("Disambiguation method: " + self.ctx_method)
        print("Filter context: %s" % (self.filter_ctx))
        
    def get_context(self, text, target_position):
        """ returns a list of words surrounding the target positioned at [start:end] in the text 
        target_pos is a string 'start,end' 
        window=5 means 5 words on the left + 5 words on the right are returned, if they exist"""
        start, end = [int(x) for x in target_position.split(',')]
        l, r = text[:start].split(), text[end:].split()
    
        # it only makes sense to use context for which we have vectors
        l = [ctx for ctx in l if ctx in self.vc.vocab]
        r = [ctx for ctx in r if ctx in self.vc.vocab]
        
        # filter polysemous words from context
        # if self.filter_ctx:
        #    l = [ctx for ctx in l if len(get_senses(self.vs, ctx)) < 2] 
        #    r = [ctx for ctx in r if len(get_senses(self.vs, ctx)) < 2]
            
        return l[-self.window:] + r[:self.window]
        
        # TODO: clear test example from stop words?
    
    # NOTE: if model is loaded with norm_only=True (that's default), then
    # both syn0 and syn0norm contain normalized vectors. 
    # In this case model['word'] shortcut also returns a normalized vector
    def __logprob__(self, cv, vsense):
        """ returns P(vsense|cv), where vsense is a vector, cv is a context vector """
        return 1.0 / (1.0 + np.exp(-np.dot(cv, vsense)))
        
    def __prob_sep__(self, vctx, vsense):
        """ returns probability of a sense (vector) in a given context (list of context word vectors) using 'sep' method"""
        return reduce(mul, [self.__logprob__(cv, vsense) for cv in vctx], 1)
    
    def __cosine_sim__(self, v1, v2):
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)
    
    def __filter_sim__(self, vctx, senses, n=2):
        """ returns n most relevant for WSD context vectors (for cosine similarity approach) """
        prob_dist_per_cv = [[self.__cosine_sim__(cv, self.vs[sense]) for sense in senses] for cv in vctx]
        significance = [abs(max(pd) - min(pd)) for pd in prob_dist_per_cv]
        most_significant_cv = sorted(zip(vctx, significance), key = itemgetter(1), reverse=True)[:n]
        
        return [cv for cv, sign in most_significant_cv]
        

    def norm(self, a):
        s = sum(a)
        return [float(p)/s for p in a]

    def entropy(self, dist):
        # give some probability to zero elements
        print "Distribution for senses: ", dist
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
    
    def __dis_context__(self, context, word):
        """ disambiguates the sense of a word for a given list of context words
            context - a list of context words
            word  - word to be disambiguated
            returns None if word is not covered by the model"""
        senses = get_senses(self.vs, word)
        if len(senses) == 0: # means we don't know any sense for this word
            return None 
        
        # collect context vectors
        vctx = [self.vc[c] for c in context]
       
        if len(vctx) == 0: # means we have no context to rely on
            return None
        # TODO: better return most frequent sense or make random choice
        
        if self.ctx_method == 'sep':
            prob_dist = [self.__prob_sep__(vctx, self.vs[sense]) for sense in senses]
        
        elif self.ctx_method == 'avg':
            avg_context = np.mean(vctx, axis=0)
            prob_dist = [self.__logprob__(avg_context, self.vs[sense]) for sense in senses]
        elif self.ctx_method == 'sim':
            if self.filter_ctx:
                vctx = self.__filter_sim__(vctx, senses)
            avg_context = np.mean(vctx, axis=0)
            prob_dist = [self.__cosine_sim__(avg_context, self.vs[sense]) for sense in senses]
        else:
            raise ValueError("Unknown context handling method '%s'" % self.ctx_method) 
            
        
        #try:
        #    e_confidence = self.entropy(prob_dist)
        #except: 
        #    print word, senses, vctx, prob_dist
        e_confidence = None# self.entropy(prob_dist)
        diff_confidence = None #self.diff_confidence(prob_dist)
        # return sense (word#id), probability, entropy confidence, differences confidence, prob_dist and length of context
        return senses[np.argmax(prob_dist)], prob_dist, e_confidence, diff_confidence, len(context)
    
    def dis_text(self, text, pos, word):
        """ disambiguates the sense of a word in given text
            text - a tokenized string ("Obviously , it was cold .")
            pos - position of a word in text ("12,14")
            word  - word to be disambiguated ("it")
            returns None if word is not covered by the model"""
        ctx = self.get_context(text, pos)
        return self.__dis_context__(ctx, word)


# Example:
# text = "However , the term mouse can also be applied to species outside of this genus . Mouse often refers to any small muroid rodent , while rat refers to larger muroid rodents"
# pos = "80,85"
# word = "mouse"

# dis_context follows approach: window first, filter after
# dis_text follows approach: filter first, window later
# TODO: find consistent approach!

        
    