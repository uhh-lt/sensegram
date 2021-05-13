from vector_representations.sense_vectors import SenseVectors
from collections import defaultdict
import codecs
import joblib
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm


class SparseSenseVectors(SenseVectors):
    """ Class that stores information about sense vectors """

    def __init__(self, pcz_fpath, word_vectors_obj=None, save_pkl=True, sense_dim_num=1000,
                 norm_type="sum", weight_type="score", max_cluster_words=20):
        super(SparseSenseVectors, self).__init__(pcz_fpath, word_vectors_obj, save_pkl, sense_dim_num,
            norm_type, weight_type, max_cluster_words)

    def _load_sense2vector_precomp(self, sense2vector_fpath):
        return joblib.load(sense2vector_fpath)

    def get_senses(self, word_i, ignore_case=False, generate_pos=False):

        if ignore_case: words = self._generate_mixed_cases(word_i)
        else: words = [word_i]
        senses_nopos = self._retrieve_senses(words)

        if generate_pos:
            words_pos = []
            for pos in self.POS:
                for w in words:
                    words_pos.append(w + self.SEP_SENSE_POS + pos)
            senses_pos = self._retrieve_senses(words_pos)
        else:
            senses_pos = []

        return senses_nopos + senses_pos

    def _retrieve_senses(self, words):
        senses = []
        for w in words:
            if w in self.sense_vectors:
                for sense_id in self.sense_vectors[w]:
                    prob = self.pcz.get_sense_prob(w, sense_id)
                    senses.append((w, sense_id, prob))
        return senses

    def similarity(self, word_i, sense_i, word_j, sense_j, use_word_vectors=False, unit_norm=False):
        oov = word_i not in self.sense_vectors or \
              word_j not in self.sense_vectors or \
              sense_i not in self.sense_vectors[word_i] or \
              sense_j not in self.sense_vectors[word_j]
        if oov:
            print("Warning: out of vocabulary:", word_i, sense_i, word_j, sense_j)
            return 0.0

        sense_vector_i = self.sense_vectors[word_i][sense_i]
        sense_vector_j = self.sense_vectors[word_j][sense_j]
        if unit_norm:
            sense_vector_i = sense_vector_i / norm(sense_vector_i)
            sense_vector_j = sense_vector_j / norm(sense_vector_j)

        if use_word_vectors:
            if self.word_vectors is not None:
                sense_vector_i = self._mixing(sense_vector_i, word_i)
                sense_vector_j = self._mixing(sense_vector_j, word_j)
            else:
                print("Warning: cannot use word vectors as they were not loaded.")

        s = sense_vector_i.dot(sense_vector_j.T) # assuming that vectors are unit norm for cosine
        if s != 0: return s.data[0]
        else: return 0.0

    def _mixing(self, sense_vector_i, word_i):
        if word_i in self.word_vectors.word2idx:
            vector_i = self.word_vectors.vectors[self.word_vectors.word2idx[word_i]]
            vector_i = vector_i / norm(vector_i) # too scary to sum up w/o any normalization
            sense_vector_i = sense_vector_i / norm(sense_vector_i)
            sense_vector_i = self.SENSE_WEIGHT*sense_vector_i + self.WORD_WEIGHT*vector_i
        else:
            print("Warning: mixing not possible, word '%s' not found" % word_i)

        return sense_vector_i

    def build(self,
              wv,  # wvo = an instance of WordVectors
              sense_dim_num=10000,
              save_pkl=True,
              norm_type="sum",
              weight_type="ones",
              max_cluster_words=20):
        """
        Build sense vectors out of sparse word_vectors and save them in csv and binary formats.
        The latter requires storing all vectors in memory, the former has no such requirement.
        weight_type in "ones", "score", "rank"
        """

        if save_pkl: sense2vector = defaultdict() # word -> sense_id -> sparse_vector

        with codecs.open(self.sense_vectors_csv_fpath, "w", "utf-8") as csv_file:
            print("word\tcid\tcluster\tisas\tfeatures", file=csv_file)
            sense_count = 0
            for word in self.pcz.data:
                for sense_id in self.pcz.data[word]:
                    sense_count += 1
                    if sense_count % 10000 == 0: print(sense_count, "senses processed")
                    sense_vector = csr_matrix(wv.vectors[0].shape)
                    for i, cluster_word in enumerate(self.pcz.data[word][sense_id]["cluster"]):
                        if cluster_word in wv.word2idx:
                            cw = cluster_word
                        else:
                            f = cluster_word.split(self.SEP_SENSE_POS)
                            if len(f) == 2 and f[0] in wv.word2idx:
                                cw = f[0]
                            elif len(f) == 3 and ((f[0] + self.SEP_SENSE_POS + f[1]) in wv.word2idx):
                                cw = f[0] + self.SEP_SENSE_POS + f[1]
                            else:
                                if self.VERBOSE: print("Warning: cluster word '%s' is OOV." % (cluster_word))
                                continue

                        if weight_type == "ones": weight = 1.0
                        elif weight_type == "score": weight = self.pcz.data[word][sense_id]["cluster"][cluster_word]
                        elif weight_type == "rank": weight = 1.0/(i+1)
                        else: weight = self.pcz.data[word][sense_id]["cluster"][cluster_word]

                        sense_vector += weight * wv.vectors[wv.word2idx[cw]]

                    normalizer = self._normalizer(word, sense_id, norm_type, weight_type, max_cluster_words)
                    sense_vector = sense_vector / normalizer

                    if save_pkl:
                        if word not in sense2vector: sense2vector[word] = {}
                        sense2vector[word][sense_id] = sense_vector

                    # sort the result by value
                    indicies = sense_vector.data.argsort()[::-1][:sense_dim_num]
                    sense_features = []
                    for idx in indicies:
                        sense_features.append("%s:%.2f" % (wv.features[sense_vector.indices[idx]],
                                                           sense_vector.data[idx]))

                    # save feature representation as string
                    features_str = ", ".join(sense_features)
                    cluster_str = ", ".join("%s:%.3f" % (w, self.pcz.data[word][sense_id]["cluster"][w]) for w in
                                            self.pcz.data[word][sense_id]["cluster"])
                    hypers_str = ", ".join(
                        "%s:%.3f" % (w, self.pcz.data[word][sense_id]["isas"][w]) for w in self.pcz.data[word][sense_id]["isas"])
                    print("%s\t%s\t%s\t%s\t%s" % (
                        word,
                        str(sense_id),
                        cluster_str,
                        hypers_str,
                        features_str), file=csv_file)

        if save_pkl:
            joblib.dump(sense2vector, self.sense_vectors_bin_fpath)
            print("Dictionary of vectors:", self.sense_vectors_bin_fpath)

        print("Created %d sense vectors" % sense_count)

        return sense2vector
