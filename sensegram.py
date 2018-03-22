import os.path
import codecs
import numpy as np
from gensim.models import word2vec
from collections import defaultdict
from traceback import format_exc
import gensim


DEFAULT_COUNT = 100 # Should be larger than min_count of vec object, which is 5 by default
SEP_SENSE = "#"
INVENTORY_EXT = ".inventory.csv"


class SenseGram(word2vec.Word2Vec):

    def __init__(self, *args, **kwargs):
        super(SenseGram, self).__init__(*args, **kwargs)
        self.inventory = defaultdict(lambda: defaultdict(float))

    def max_pairwise_sim(self, word_i, word_j, ignore_case=False):
        """ Calculates maximal pairwise similarity between all senses. """

        senses_word_i = [s for s, p in self.get_senses(word_i, ignore_case=ignore_case)]
        senses_word_j = [s for s, p in self.get_senses(word_j, ignore_case=ignore_case)]
        sims_ij = []
        for n, s_i in enumerate(senses_word_i):
            for m, s_j in enumerate(senses_word_j):
                sims_ij.append((self.similarity(s_i, s_j), s_i, s_j))

        sims_ij = sorted(sims_ij, reverse=True)
        if len(sims_ij) > 0:
            max_sim_ij = sims_ij[0][0]
            return max_sim_ij
        else:
            return 0.0

    def create_zero_vectors(self, senses_num, vector_dim):
        """ Resets existing word vectors and creates new vectors.
         This is useful if you try to create a model from scratch. """

        self.wv.syn0 = np.zeros((senses_num, vector_dim), dtype=np.float32)

    def get_senses(self, word, ignore_case=False):
        """ Returns a list of all available senses for a given word.
        example: 'mouse' -> [('mouse#0', 0.33), ('mouse#1', 0.66)] """

        words = set([word])
        senses = []
        if ignore_case:
            words.add(word.title())
            words.add(word.lower())
        
        for word in words:
            if word not in self.inventory: continue
            for sense_id in self.inventory[word]:
                sense = word + SEP_SENSE + str(sense_id)
                if sense not in self.wv.vocab: continue
                senses.append((sense, self.inventory[word][sense_id]))
        return senses
 
    def get_most_probable_sense(self, word, ignore_case=True):
        senses = self.get_senses(word, ignore_case=ignore_case)
        most_probable_sense, prob = sorted(senses, key=lambda s: s[1], reverse=True)[0]
        return most_probable_sense, prob
   
    def save_word2vec_format(self, fname, fvocab=None, binary=False):
        """ Saves SenseGram model in the word2vec format. In addition a CSV
        file with word sense inventory is saved containing a priory probabilities."""

        # Save the word2vec format model
        self.wv.save_word2vec_format(fname, fvocab, binary)

        # Save the extra file 'word#sense_id<TAB>prob-of-the-sense' with sense inventory info
        prob_file = fname + INVENTORY_EXT
        with codecs.open(prob_file, 'w', encoding='utf-8') as out:
            for word in self.inventory:
                for sense_id in self.inventory[word]:
                    out.write("%s#%s\t%.6f\n" % (word, sense_id, self.inventory[word][sense_id]))
    
    @classmethod
    def load_word2vec_format(cls, model_fpath, fvocab=None, binary=False, norm_only=True, encoding='utf8', unicode_errors='strict'):
        """ Load the model from word2vec format (the vectors) and optionally loads word sense inventory
        from a CSV file located next to the word vectors. """

        # Load word vectors
        wv_obj = gensim.models.KeyedVectors.load_word2vec_format(model_fpath, fvocab, binary, encoding, unicode_errors)
        result = cls(size=wv_obj.syn0.shape[1])
        result.wv.syn0 = wv_obj.syn0
        result.wv.vocab = wv_obj.vocab
        result.wv.index2word = wv_obj.index2word

        # Load the inventory
        inventory_fpath = model_fpath + INVENTORY_EXT
        if os.path.isfile(inventory_fpath):
            with codecs.open(inventory_fpath, 'r', encoding='utf-8') as inventory_file:
                for line in inventory_file:
                    try:
                        sense, prob = line.split('\t')
                        f = sense.split(SEP_SENSE)
                        word = SEP_SENSE.join(f[0:len(f)-1])  # some words can contains sep
                        sense_id = f[-1]
                        if len(word) == 0 or len(sense_id) == 0: continue
                        result.inventory[word][sense_id] = float(prob)
                    except:
                        print(("Bad line '%s'" % line))
                        print((format_exc()))
        else:
            for sense in result.wv.index2word:
                try:
                    word, sense_id = sense.split(SEP_SENSE)
                    result.inventory[word][sense_id] = 1.0
                except:
                    print(format_exc())
                    
        return result
        
    def add_sense(self, word, sense_id, vector, prob):
        """ Add a new sense to the model, where sense is an
        identifier composed composed of a word and an integer sense id, e.g. 'python#2'.
        The vector is a regular word2vec vector in the form of ndarray.
        The prob is a priory probability of the word sense among all senses of the word,
        e.g. "python#1" is 0.33 and "python#2" is 0.67. """

        # Update the word2vec model: vector and word2vec vocabulary
        if hasattr(self.wv, 'syn0'):
            word_id = len(self.wv.vocab)

            sense = word.replace(" ","_") + SEP_SENSE + str(sense_id) # w2v format accepts no whitespaces
            self.wv.vocab[sense] = word2vec.Vocab(index=word_id, count=DEFAULT_COUNT)
            self.wv.syn0[word_id] = vector
            self.wv.index2word.append(sense)
            assert sense == self.wv.index2word[self.wv.vocab[sense].index]
        else: 
            raise RuntimeError("Error: you should initialize syn0 matrix before adding words")

        # Update the custom word sense inventory
        self.inventory[word][sense_id] = prob

