from operator import itemgetter
import os.path
import codecs
import math
import numpy as np
from gensim.models import word2vec

default_count = 100 # arbitrary, should be larger than min_count of vec object, which is 5 by default

class Sense2Vec(word2vec.Word2Vec):
    def __init__(self, *args, **kwargs):
        super(Sense2Vec, self).__init__(*args, **kwargs)
        self.probs = {} # mapping from a sense (String) to its probability
    
    def get_senses(self, word, ignore_case=False):
        """ returns a list of all available senses for a given word.
        example: 'mouse' -> ['mouse#0', 'mouse#1', 'mouse#2']
        Assumption: senses use continuous numbering"""
        words = [word]
        senses = []
        if ignore_case:
            words.append(word[0].upper() + word[1:])
            words.append(word[0].lower() + word[1:])
        
        words = set(words)
        for word in words:
            for i in range(0,200):
                sense = word + u'#' + unicode(i)
                if sense in self.vocab:
                    senses.append((sense, self.probs[sense]))
                else:
                    break
        return senses
    
    def save_word2vec_format(self, fname, fvocab=None, binary=False):
        super(Sense2Vec, self).save_word2vec_format(fname, fvocab, binary)
        
        prob_file = fname + ".probs"
        with codecs.open(prob_file, 'w', encoding='utf-8') as out:
            for sense, prob in self.probs.items():
                out.write("%s %s\n" % (sense, prob))
    
    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, norm_only=True, encoding='utf8', unicode_errors='strict'):
        mod = word2vec.Word2Vec.load_word2vec_format(fname, fvocab, binary, norm_only, encoding, unicode_errors)
        
        result = cls(size=mod.vector_size)
        result.syn0 = mod.syn0
        result.vocab = mod.vocab
        result.index2word = mod.index2word
        
        prob_file = fname + ".probs"
        if os.path.isfile(prob_file):
            with codecs.open(prob_file, 'r', encoding='utf-8') as inp: 
                for line in inp:
                    sense, prob = line.split()
                    result.probs[sense] = float(prob)
                    
        return result
        
    def add_word(self, word, vector):
        """add new word to the model"""
        if hasattr(self, 'syn0'):
            word_id = len(self.vocab)
            self.vocab[word] = word2vec.Vocab(index=word_id, count=default_count)
            self.syn0[word_id] = vector
            self.index2word.append(word)

            assert word == self.index2word[self.vocab[word].index]
        else: 
            raise RuntimeError("must initialize syn0 matrix before adding words")
        
    def __normalize_probs__(self, cluster_sum):
        for sense, cluster_size in self.probs.items():
            word, sense_id = sense.split("#")
            if word in cluster_sum and cluster_sum[word] > 0:
                self.probs[sense] = float(cluster_size)/cluster_sum[word]
            else:
                self_probs[sense] = 1

class WSD(object):
    
    def __init__(self, path_to_sense_model, path_to_context_model, window=10, method="sim", filter_ctx=2, ignore_case=False):
        self.vs = Sense2Vec.load_word2vec_format(path_to_sense_model, binary=True)
        self.vc = word2vec.Word2Vec.load_word2vec_format(path_to_context_model, binary=True)
        self.window = window
        self.ctx_method = method
        self.filter_ctx = filter_ctx
        self.ignore_case = ignore_case
        
        print("Disambiguation method: " + self.ctx_method)
        print("Filter context: f = %s" % (self.filter_ctx))
        
    def get_context(self, text, start, end):
        """ returns a list of words surrounding the target positioned at [start:end] in the text 
        target_pos is a string 'start,end' 
        window=5 means 5 words on the left + 5 words on the right are returned, if they exist"""
        
        l, r = text[:start].split(), text[end:].split()
    
        # it only makes sense to use context for which we have vectors
        l = [ctx for ctx in l if ctx in self.vc.vocab]
        r = [ctx for ctx in r if ctx in self.vc.vocab]
            
        return l[-self.window:] + r[:self.window]
    
    def __logprob__(self, cv, vsense):
        """ returns P(vsense|cv), where vsense is a vector, cv is a context vector """
        return 1.0 / (1.0 + np.exp(-np.dot(cv, vsense)))
    
    def __cosine_sim__(self, v1, v2):
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)
    
    def __filter__(self, vctx, senses, n):
        """ returns n most relevant for WSD context vectors """
        
        if self.ctx_method == 'prob':
            prob_dist_per_cv = [[self.__logprob__(cv, self.vs[sense]) for sense, prob in senses] for cv in vctx]
        elif self.ctx_method == 'sim':
            prob_dist_per_cv = [[self.__cosine_sim__(cv, self.vs[sense]) for sense, prob in senses] for cv in vctx]
        else:
            raise ValueError("Unknown context handling method '%s'" % self.ctx_method)
            
        significance = [abs(max(pd) - min(pd)) for pd in prob_dist_per_cv]
        most_significant_cv = sorted(zip(vctx, significance), key = itemgetter(1), reverse=True)[:n]
        
        return [cv for cv, sign in most_significant_cv]
    
    def __dis_context__(self, context, word):
        """ disambiguates the sense of a word for a given list of context words
            context - a list of context words
            word  - word to be disambiguated
            returns None if word is not covered by the model"""
        senses = self.vs.get_senses(word, self.ignore_case)
        if len(senses) == 0: # means we don't know any sense for this word
            return None 
        
        # collect context vectors
        vctx = [self.vc[c] for c in context]
       
        if len(vctx) == 0: # means we have no context
            return None
        # TODO: better return most frequent sense or make random choice
        
        # filter context vectors, if aplicable
        if self.filter_ctx >= 0:
                vctx = self.__filter__(vctx, senses, self.filter_ctx)
        
        if self.ctx_method == 'prob':
            avg_context = np.mean(vctx, axis=0)
            scores = [self.__logprob__(avg_context, self.vs[sense]) for sense, prob in senses]
            
        elif self.ctx_method == 'sim':
            avg_context = np.mean(vctx, axis=0)
            scores = [self.__cosine_sim__(avg_context, self.vs[sense]) for sense, prob in senses]
            
        else:
            raise ValueError("Unknown context handling method '%s'" % self.ctx_method) 
        
        # return sense (word#id), scores for senses
        return senses[np.argmax(scores)], scores
    
    def dis_text(self, text, target, target_start, target_end):
        """ disambiguates the sense of a word in given text
            text - a tokenized string ("Obviously , cats are funny .")
            target  - a target word (lemma) to be disambiguated ("cat")
            target_start - start index of target word occurence in text (12)
            target_end - end index of target word occurence in text, considering flexed forms (16)
            
            returns None if word is not covered by the model"""
        
        ctx = self.get_context(text, target_start, target_end)
        return self.__dis_context__(ctx, target)