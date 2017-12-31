from operator import itemgetter
import math
import numpy as np


class WSD(object):
    """ Object that for word sense disambiguation. """

    def __init__(self, vs, vc, window=10, method="sim", filter_ctx=2, ignore_case=False, verbose=False):
        self.vs = vs
        self.vc = vc
        self.window = window
        self.ctx_method = method
        self.filter_ctx = filter_ctx
        self.ignore_case = ignore_case
        self.verbose = verbose

        print(("Disambiguation method: " + self.ctx_method))
        print(("Filter context: f = %s" % (self.filter_ctx)))

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
            x = v1[i];
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        return sumxy / math.sqrt(sumxx * sumyy)

    def __filter__(self, vctx, senses, n):
        """ returns n most relevant for WSD context vectors """

        if self.ctx_method == 'prob':
            prob_dist_per_cv = [[self.__logprob__(cv, self.vs[sense]) for sense, prob in senses] for cv in vctx]
        elif self.ctx_method == 'sim':
            prob_dist_per_cv = [[self.__cosine_sim__(cv, self.vs[sense]) for sense, prob in senses] for cv in vctx]
        else:
            raise ValueError("Unknown context handling method '%s'" % self.ctx_method)

        significance = [abs(max(pd) - min(pd)) for pd in prob_dist_per_cv]
        if self.verbose:
            print("Significance scores of context words:")
            print(significance)
        most_significant_cv = sorted(zip(vctx, significance), key=itemgetter(1), reverse=True)[:n]

        return [cv for cv, sign in most_significant_cv]

    def __dis_context__(self, context, word):
        """ disambiguates the sense of a word for a given list of context words
            context - a list of context words
            word  - word to be disambiguated
            returns None if word is not covered by the model"""
        senses = self.vs.get_senses(word, self.ignore_case)
        if self.verbose:
            print("Senses of a target word:")
            print(senses)

        if len(senses) == 0:  # means we don't know any sense for this word
            return None

            # collect context vectors
        vctx = [self.vc[c] for c in context]

        if len(vctx) == 0:  # means we have no context
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
            if self.verbose:
                print("Sense probabilities:")
                print(scores)

        else:
            raise ValueError("Unknown context handling method '%s'" % self.ctx_method)

            # return sense (word#id), scores for senses
        return senses[np.argmax(scores)][0], scores

    def dis_text(self, text, target, target_start, target_end):
        """ disambiguates the sense of a word in given text
            text - a tokenized string ("Obviously , cats are funny .")
            target  - a target word (lemma) to be disambiguated ("cat")
            target_start - start index of target word occurence in text (12)
            target_end - end index of target word occurence in text, considering flexed forms (16)

            returns None if word is not covered by the model"""

        ctx = self.get_context(text, target_start, target_end)
        if self.verbose:
            print("Extracted context words:")
            print(ctx)
        return self.__dis_context__(ctx, target)