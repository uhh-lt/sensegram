from pcz.sense_clusters import SenseClusters
from os.path import exists

D = False # debug
V = False # verbose

def generate_mixed_cases(word, full_upper=False, generate_pos=False):
    return SenseVectors._generate_mixed_cases(word, full_upper, generate_pos)


class SenseVectors(object):
    """ Class that stores information about sense vectors """

    VECTORS_CSV_EXT = ".sense_vectors.csv"
    VECTORS_BIN_EXT = ".sense_vectors"  
    SEP_SENSE_POS = "#"
    POS = ["NN", "NP", "JJ", "VB"]
    SENSE_WEIGHT = 0.5
    WORD_WEIGHT = 0.5
    VERBOSE = True

    def __init__(self, pcz_fpath, word_vectors_obj=None, save_pkl=True, sense_dim_num=10000,
                 norm_type="sum", weight_type="score", max_cluster_words=20):
        self.pcz_fpath = pcz_fpath
        self.params = "-".join([str(sense_dim_num), norm_type, weight_type, str(max_cluster_words)])
        self.sense_vectors_bin_fpath = self.pcz_fpath + "-" + self.params + self.VECTORS_BIN_EXT
        self.sense_vectors_csv_fpath = self.pcz_fpath + "-" + self.params + self.VECTORS_CSV_EXT
        self.word_vectors = word_vectors_obj

        if exists(pcz_fpath):
            self.pcz = SenseClusters(pcz_fpath, strip_dst_senses=False, load_sim=True, verbose=False)
            self.sense_vectors = self.load(self.sense_vectors_bin_fpath)
            if self.sense_vectors == None:
                if self.word_vectors != None:
                    print("No pre-calculated model found at:", self.sense_vectors_bin_fpath)
                    print("Building a new model from:", self.pcz_fpath)
                    self.sense_vectors = self.build(self.word_vectors,
                               sense_dim_num=sense_dim_num,
                               save_pkl=save_pkl,
                               norm_type=norm_type,
                               weight_type=weight_type,
                               max_cluster_words=max_cluster_words)
            else:
                print("Loaded model from:", pcz_fpath)
        else:
            print("File not found:", pcz_fpath)

    def get_senses(self, word_i, ignore_case=False):
        return []

    def similarity(self, word_i, sense_i, word_j, sense_j, use_word_vectors=False):
        return 0.0

    @classmethod
    def _generate_mixed_cases(self, word, full_upper=False, generate_pos=False):
        """ For a single word, generates its lower cased and capitalized versions."""

        if full_upper:
            words = [word.lower(), word.upper()]
        else:
            words = [word.lower()]

        if len(word) > 1:
            words.append(word[0].upper() + word[1:].lower())

        if generate_pos:
            words_pos = []
            for w in words:
                for p in self.POS:
                    words_pos.append(w + self.SEP_SENSE_POS + p)
            return words_pos
        else:
            return words

    def max_similarity_pos(self, word_i, word_j, ignore_case=False, unit_norm=False, use_word_vectors=False,):
        words_i_pos = self._generate_mixed_cases(word_i, generate_pos=True)
        words_j_pos = self._generate_mixed_cases(word_j, generate_pos=True)

        sims_ij = []
        for w_i in words_i_pos:
            for w_j in words_j_pos:
                sims_ij.append((self.max_pairwise_sim(
                    w_i, w_j,ignore_case=ignore_case, unit_norm=unit_norm,
                    use_word_vectors=use_word_vectors), w_i, w_j))

        sims_ij = sorted(sims_ij, reverse=True)
        if len(sims_ij) > 0:
            max_sim_ij = sims_ij[0][0]
            return max_sim_ij
        else:
            return 0.0

    def max_pairwise_sim(self, word_i, word_j, ignore_case=False, unit_norm=False, use_word_vectors=False):
        """ Calculates maximal pairwise similarity between all senses. """

        senses_word_i = [(w, s) for w, s, p in self.get_senses(word_i, ignore_case=ignore_case)]
        senses_word_j = [(w, s) for w, s, p in self.get_senses(word_j, ignore_case=ignore_case)]
        sims_ij = []
        for w_i, s_i in senses_word_i:
            for w_j, s_j in senses_word_j:
                sims_ij.append((self.similarity(
                    w_i, s_i, w_j, s_j, unit_norm=unit_norm,
                    use_word_vectors=use_word_vectors), s_i, s_j))

        sims_ij = sorted(sims_ij, reverse=True)
        if len(sims_ij) > 0:
            max_sim_ij = sims_ij[0][0]
            return max_sim_ij
        else:
            return 0.0

    def _load_sense2vector_precomp(self, sense2vector_fpath):
        return None

    def _normalizer(self, word, sense_id, norm_type, weight_type, max_cluster_words):
        """ Calculates normalizer for a sense cluster (sum/weighted sum of vectors). """
        if norm_type == "sum":
            # do normalize
            if weight_type == "ones":
                normalizer = min(max_cluster_words, len(self.pcz.data[word][sense_id]["cluster"]))
            elif weight_type == "score":
                normalizer = 0.0
                for i, w in enumerate(self.pcz.data[word][sense_id]["cluster"]):
                    if i >= max_cluster_words: break
                    normalizer += self.pcz.data[word][sense_id]["cluster"][w]
            elif weight_type == "rank":
                normalizer = 0.0
                for i, x in enumerate(range(len(self.pcz.data[word][sense_id]["cluster"]))):
                    if i >= max_cluster_words: break
                    normalizer += x + 1
            else:
                normalizer = 1.0
        else:
            # do not normalize
            normalizer = 1.0

        if normalizer > 0.0: return normalizer
        else: return 1.0

    def load(self, sense2vector_fpath):
        if exists(sense2vector_fpath):
            sense2vector = self._load_sense2vector_precomp(sense2vector_fpath)
            print("Loaded a pre-computed model from:", sense2vector_fpath)
        else:
            print("Cannot load a pre-computed model from:", sense2vector_fpath)
            sense2vector = None

        return sense2vector

    def build(self,
              wv, # an instance of some word vectors object
              sense_dim_num=10000,
              save_pkl=True,
              norm_type="sum"):
        return
