from operator import itemgetter
import math
import numpy as np
import stop_words
   

class WSD(object):
    """ Object that for word sense disambiguation. """

    def __init__(self, vs, vc, window=10, method="sim", lang="en", filter_ctx=3, ignore_case=False, verbose=False):
        self._vs = vs
        self._vc = vc
        self._window = window
        self._ctx_method = method
        self._filter_ctx = filter_ctx
        self._ignore_case = ignore_case
        self._verbose = verbose
        self._stop_words = self._get_stop_words(lang)

        print(("Disambiguation method: " + self._ctx_method))
        print(("Filter context: f = %s" % (self._filter_ctx)))

    def _get_stop_words(self, lang="en"):
        _stop_words = set()
        for sw in stop_words.get_stop_words(lang): 
            _stop_words.add(sw)
            _stop_words.add(sw.title())
            _stop_words.add(sw.upper())

        return _stop_words

    def get_context(self, text, start, end):
        """ returns a list of words surrounding the target positioned at [start:end] in the text
        target_pos is a string 'start,end'
        window=5 means 5 words on the left + 5 words on the right are returned, if they exist"""

        l, r = text[:start].split(), text[end:].split()

        # it only makes sense to use context for which we have vectors
        l = [ctx for ctx in l if ctx in self._vc.vocab]
        r = [ctx for ctx in r if ctx in self._vc.vocab]
        lr = set(l[-self._window:] + r[:self._window])

        return list(lr.difference(self._stop_words))

    def __logprob__(self, cv, vsense):
        """ returns P(vsense|cv), where vsense is a vector, cv is a context vector """
        return 1.0 / (1.0 + np.exp(-np.dot(cv, vsense)))

    def __cosine_sim__(self, v1, v2):
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i];
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        return sumxy / math.sqrt(sumxx * sumyy)

    def __filter__(self, vctx, senses, n, context):
        """ returns n most relevant for WSD context vectors """

        if self._ctx_method == 'prob':
            prob_dist_per_cv = [[self.__logprob__(cv, self._vs[sense]) for sense, prob in senses] for cv in vctx]
        elif self._ctx_method == 'sim':
            prob_dist_per_cv = [[self.__cosine_sim__(cv, self._vs[sense]) for sense, prob in senses] for cv in vctx]
        else:
            raise ValueError("Unknown context handling method '%s'" % self._ctx_method)

        significance = [abs(max(pd) - min(pd)) for pd in prob_dist_per_cv]
        if self._verbose:
            print("Significance scores of context words:")
            print(significance)
        most_significant_cv = sorted(zip(vctx, significance, context), key=itemgetter(1), reverse=True)[:n]
        
        if self._verbose: 
            print("Context words:")
            for _, sign, word in most_significant_cv: print("{}\t{:.3f}".format(word, sign))

        context_vectors = [cv for cv, sign, word in most_significant_cv]
        
        return context_vectors

    def _disambiguate_context(self, context, word, all_cases):
        """ disambiguates the sense of a word for a given list of context words
            context - a list of context words
            word  - word to be disambiguated
            returns None if word is not covered by the model"""
       
        senses = self._vs.get_senses(word, self._ignore_case)

        if self._verbose:
            print("Senses of a target word:")
            print(senses)

        if len(senses) == 0:  # means we don't know any sense for this word
            return senses, []

        # collect context vectors
        vctx = [self._vc[c] for c in context]

        if len(vctx) == 0:  # means we have no context
            mfs_sense_id, mfs_prob = self._vc.get_most_probable_sense(word, self._ignore_case)
            return mfs_sense_id, [mfs_prob]

        # filter context vectors, if aplicable
        if self._filter_ctx >= 0:
            vctx = self.__filter__(vctx, senses, self._filter_ctx, context)

        if self._ctx_method == 'prob':
            avg_context = np.mean(vctx, axis=0)
            scores = [self.__logprob__(avg_context, self._vs[sense]) for sense, prob in senses]

        elif self._ctx_method == 'sim':
            avg_context = np.mean(vctx, axis=0)
            scores = [self.__cosine_sim__(avg_context, self._vs[sense]) for sense, prob in senses]
            if self._verbose:
                print("Sense probabilities:")
                print(scores)

        else:
            raise ValueError("Unknown context handling method '%s'" % self._ctx_method)

            # return sense (word#id), scores for senses
        return senses[np.argmax(scores)][0], scores

    def disambiguate(self, context, word, all_cases=False):
        begin_index = context.find(word)
        end_index = begin_index + len(word)
        return self._disambiguate(context, word, begin_index, end_index, all_cases)
        
    def _disambiguate(self, text, target, target_start, target_end, all_cases=False):
        """ disambiguates the sense of a word in given text
            text - a tokenized string ("Obviously , cats are funny .")
            target  - a target word (lemma) to be disambiguated ("cat")
            target_start - start index of target word occurence in text (12)
            target_end - end index of target word occurence in text, considering flexed forms (16)
            returns None if word is not covered by the model """

        ctx = self.get_context(text, target_start, target_end)
        if self._verbose:
            print("Extracted context words:")
            print(ctx)
        return self._disambiguate_context(ctx, target, all_cases)

