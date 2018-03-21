from operator import itemgetter
import math
import numpy as np
import stop_words
   

class WSD(object):
    """ Object that for word sense disambiguation. """

    def __init__(self, sense_vectors, word_vectors, window=10, method="sim",
            lang="en", max_context_words=3, ignore_case=False, verbose=False):

        self._sense_vectors = sense_vectors
        self._word_vectors = word_vectors
        self._window = window
        self._ctx_method = method
        self._max_context_words = max_context_words
        self._ignore_case = ignore_case
        self._verbose = verbose
        self._stop_words = self._get_stop_words(lang)

    def _get_stop_words(self, lang="en"):
        _stop_words = set()
        for sw in stop_words.get_stop_words(lang): 
            _stop_words.add(sw)
            _stop_words.add(sw.title())
            _stop_words.add(sw.upper())

        return _stop_words

    def get_context(self, text, start, end):
        """ Returns a list of words surrounding the target positioned at [start:end] in the text
        target_pos is a string 'start,end'
        window=5 means 5 words on the left + 5 words on the right are returned, if they exist"""

        l, r = text[:start].split(), text[end:].split()

        # it only makes sense to use context for which we have vectors
        l = [ctx for ctx in l if ctx in self._word_vectors.vocab]
        r = [ctx for ctx in r if ctx in self._word_vectors.vocab]
        lr = set(l[-self._window:] + r[:self._window])

        return list(lr.difference(self._stop_words))

    def _logprob(self, cv, vsense):
        """ Returns P(vsense|cv), where vsense is a vector, cv is a context vector """
        return 1.0 / (1.0 + np.exp(-np.dot(cv, vsense)))

    def _cos(self, v1, v2):
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i];
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        return sumxy / math.sqrt(sumxx * sumyy)

    def _filter_context(self, context_vectors, senses, n, context):
        """ Returns n most relevant for WSD context vectors """

        if self._ctx_method == 'prob':
            prob_dist_per_cv = [[self._logprob(cv, self._sense_vectors[sense]) for sense, prob in senses] for cv in context_vectors]
        elif self._ctx_method == 'sim':
            prob_dist_per_cv = [[self._cos(cv, self._sense_vectors[sense]) for sense, prob in senses] for cv in context_vectors]
        else:
            raise ValueError("Unknown context handling method '%s'" % self._ctx_method)

        significance = [abs(max(pd) - min(pd)) for pd in prob_dist_per_cv]
        if self._verbose:
            print("Significance scores of context words:")
            print(significance)
        most_significant_cv = sorted(zip(context_vectors, significance, context), key=itemgetter(1), reverse=True)[:n]
        
        if self._verbose: 
            print("Context words:")
            for _, sign, word in most_significant_cv: print("{}\t{:.3f}".format(word, sign))

        context_vectors = [cv for cv, sign, word in most_significant_cv]
        
        return context_vectors

    def _disambiguate_context(self, context, word, all_cases):
        """ Disambiguates the sense of a word for a given list of context words
            context - a list of context words
            word  - a target word to be disambiguated in the context
            returns a tuple (word#id, scores_for_senses) """
        
        senses = self._sense_vectors.get_senses(word, self._ignore_case)

        if self._verbose:
            print("Senses of a target word:")
            print(senses)

        if len(senses) == 0: 
            return "{}#0".format(word), [1.0]

        context_vectors = [self._word_vectors[c] for c in context]
        if len(context_vectors) == 0:  # means we have no context
            mfs_sense_id, mfs_prob = self._sense_vectors.get_most_probable_sense(word, self._ignore_case)
            return mfs_sense_id, [p for s, p in senses] 

        # filter context vectors, if aplicable
        if self._max_context_words >= 0:
            context_vectors = self._filter_context(context_vectors, senses, self._max_context_words, context)
        
        if self._ctx_method == 'prob':
            avg_context = np.mean(context_vectors, axis=0)
            scores = [self._logprob(avg_context, self._sense_vectors[sense]) for sense, prob in senses]
        else:
            avg_context = np.mean(context_vectors, axis=0)
            scores = [self._cos(avg_context, self._sense_vectors[sense]) for sense, prob in senses]
            if self._verbose:
                print("Sense probabilities:")
                print(scores)

        return senses[np.argmax(scores)][0], scores

    def disambiguate(self, context, word, all_cases=False):
        begin_index = context.find(word)
        end_index = begin_index + len(word)
        return self._disambiguate(context, word, begin_index, end_index, all_cases)
        
    def _disambiguate(self, text, target, target_start, target_end, all_cases=False):
        """ Disambiguates the sense of a word in given text
            text - a tokenized string ("Obviously , cats are funny .")
            target  - a target word (lemma) to be disambiguated ("cat")
            target_start - start index of target word occurence in text (12)
            target_end - end index of target word occurence in text, considering flexed forms (16) """

        ctx = self.get_context(text, target_start, target_end)
        if self._verbose:
            print("Extracted context words:")
            print(ctx)
        return self._disambiguate_context(ctx, target, all_cases)

