from vector_representations.sense_vectors import SenseVectors
import numpy as np
from traceback import format_exc
from sensegram import SenseGram
from sys import stderr


class DenseSenseVectors(SenseVectors):
    """ Class that stores information about sense vectors """

    def __init__(self, pcz_fpath, word_vectors_obj=None, save_pkl=True, sense_dim_num=10000,
            norm_type="sum", weight_type="score", max_cluster_words=20):
        super(DenseSenseVectors, self).__init__(pcz_fpath, word_vectors_obj,
            save_pkl, sense_dim_num, norm_type, weight_type, max_cluster_words)

    def _load_sense2vector_precomp(self, sense2vector_fpath):
        return SenseGram.load_word2vec_format(sense2vector_fpath)

    def get_most_probable_sense(self, word, ignore_case=True):
        senses = self.get_senses(word, ignore_case=ignore_case)
        most_probable_sense, prob = sorted(senses, key=lambda s: s[1], reverse=True)[0]
        return most_probable_sense, prob

    def get_senses(self, word_i, ignore_case=False):
        senses = []
        for word_sense, prob in self.sense_vectors.get_senses(word_i, ignore_case):
            try:
                word, sense_id = word_sense.split(self.SEP_SENSE_POS)
                senses.append( (word_i, sense_id, prob) )
            except:
                print("Wrong sense format", word_sense)
        
        return senses

    def similarity(self, word_i, sense_i, word_j, sense_j, use_word_vectors=False):
        if use_word_vectors:
            sense_vector_i = self._mixing(word_i, sense_i)
            sense_vector_j = self._mixing(word_j, sense_j)
            return sense_vector_i.dot(sense_vector_j.T)
        else:
            return self.sense_vectors.similarity(
                word_i + self.SEP_SENSE_POS + sense_i,
                word_j + self.SEP_SENSE_POS + sense_j)

    def _mixing(self, word_i, sense_i):
        if word_i in self.word_vectors.vectors:
            word_vector_i = self.word_vectors.vectors[word_i] # dense vectors are assumed to be normalized
            sense_vector_i = self.get_sense_vector(sense_i, word_i)
            sense_vector_i = self.SENSE_WEIGHT*sense_vector_i + self.WORD_WEIGHT*word_vector_i
        else:
            print("Warning: mixing not possible, word '%s' not found" % word_i)
            sense_vector_i = self.get_sense_vector(sense_i, word_i)

        return sense_vector_i

    def get_sense_vector(self, sense_i, word_i):
        sense_id = word_i + self.SEP_SENSE_POS + sense_i
        if sense_id in self.sense_vectors:
            sense_vector_i = self.sense_vectors[sense_id]
        else:
            print("Warning: sense vector not found: %s" % sense_id)
            sense_vector_i = 0.0
        return sense_vector_i

    def build(self,
              wv,  # wvo = an intance of dense word vectors
              sense_dim_num=10000,  # unused
              save_pkl=True,  # unused
              norm_type="sum",
              weight_type="score",
              max_cluster_words=20):
        """
        Build sense vectors out of word vectors and save them in binary format.
        """

        # initialize the sense vectors model
        vector_dim = wv.vectors.syn0.shape[1]
        senses_num = self.pcz.get_num_senses()
        sv = SenseGram(size=vector_dim, sorted_vocab=0)
        sv.create_zero_vectors(senses_num, vector_dim)
        sense_count = 0

        # fill the sense vectors model
        for word in self.pcz.data:
            for sense_id in self.pcz.data[word]:
                # try to build sense vector for a word sense
                try:
                    sense_count += 1
                    if sense_count % 10000 == 0: print(sense_count, "senses processed")

                    sense_vector = np.zeros(wv.vectors.syn0[0].shape, dtype=np.float32) # or the word vector?

                    non_oov = 0
                    for i, cluster_word in enumerate(self.pcz.data[word][sense_id]["cluster"]):
                        if i >= max_cluster_words: break
                        
                        # define the weight
                        if weight_type == "ones": weight = 1.0
                        elif weight_type == "score": weight = float(self.pcz.data[word][sense_id]["cluster"][cluster_word])
                        elif weight_type == "rank": weight = 1.0 / (i + 1)
                        else: weight = float(self.pcz.data[word][sense_id]["cluster"][cluster_word])
                        
                        if weight == 0:
                            print("Warning: zero weight:", cluster_word, end=' ') 
                        
                        # define the word
                        if cluster_word in wv.vectors.vocab:
                            cw = cluster_word
                        elif cluster_word.split("#")[0] in wv.vectors.vocab:
                            cw = cluster_word.split("#")[0]
                        else:
                            if self.VERBOSE:
                                print("Warning: word is OOV: '%s'" % (cluster_word), file=stderr)
                            
                            compounds = cluster_word.split("#")[0].split("_")
                            for cw in compounds:
                                if cw in wv.vectors.vocab and len(cw) > 3:
                                    if self.VERBOSE: print("Warning: adding a compound '{}' of '{}'".format(cw, cluster_word))
                                    sense_vector += (weight/len(compounds)) * wv.vectors[cw]  
                                    non_oov += 1
                            
                            continue 

                        non_oov += 1
                        sense_vector += weight * wv.vectors[cw]

                    if non_oov == 0:
                        if self.VERBOSE: print("Warning: sense is OOV: %s#%s" % (word, sense_id), file=stderr)

                    normalizer = self._normalizer(word, sense_id, norm_type, weight_type, max_cluster_words)
                    sense_vector = sense_vector / normalizer
                    sense_prob = self.pcz.get_sense_prob(word, sense_id)
                    sv.add_sense(word, sense_id, sense_vector, sense_prob)
                except:
                    print("Cannot process sense:", word, sense_id)
                    print(format_exc())

        # serialize the sense vector model
        sv.save_word2vec_format(self.sense_vectors_bin_fpath, fvocab=None, binary=False)

        print("Sense vectors:", self.sense_vectors_bin_fpath)
        print("Created %d sense vectors" % sense_count)

        return sv

