from os.path import splitext
from collections import Counter
import codecs
import gzip
from traceback import format_exc
from pprint import pprint
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
import numpy as np
from sys import stderr
from os.path import exists
from vector_representations.sense_vectors import generate_mixed_cases
from scipy.sparse.linalg import norm


class SparseWordVectors:
    """ Class that represents word vectors. """

    MATRIX_EXT = ".matrix"
    VECTORIZER_EXT = ".vectorizer.pkl"
    WORD2IDX_EXT = ".word2idx.pkl"
    FEATURES_EXT = ".features.csv"
    VERBOSE = False
    DEBUG = False

    def __init__(self, lmi_fpath=""):
        if exists(lmi_fpath):
            self.vectors, self.vectorizer, self.word2idx = self.load(lmi_fpath)
            if self.vectors == None:
                print("No model to load. Bulding a new model from:", lmi_fpath)
                self.build(lmi_fpath)
            else:
                print("Loaded model from:", lmi_fpath)
        else:
            print("File not found:", lmi_fpath)

    def load(self, lmi_fpath):
        """ Load a pre-built model from numpy files. """

        matrix_fpath = lmi_fpath + self.MATRIX_EXT
        vectorizer_fpath = lmi_fpath + self.VECTORIZER_EXT
        word2idx_fpath = lmi_fpath + self.WORD2IDX_EXT

        if exists(matrix_fpath) and exists(vectorizer_fpath) and exists(word2idx_fpath):
            word_vectors = joblib.load(matrix_fpath)
            vectorizer = joblib.load(vectorizer_fpath)
            word2idx = joblib.load(word2idx_fpath)
            print("Loaded word vectors from:", lmi_fpath)
        else:
            print("Some input files are missing. Cannot load the model.")
            print(exists(matrix_fpath), matrix_fpath)
            print(exists(vectorizer_fpath), vectorizer_fpath)
            print(exists(word2idx_fpath), word2idx_fpath)
            word_vectors = None
            vectorizer = None
            word2idx = None

        return word_vectors, vectorizer, word2idx

    def build(self, lmi_fpath):
        """ Build a new model from CSV file. """

        lmi_word2idx = {}
        lmi_list = []

        is_gzip = splitext(lmi_fpath)[-1] == ".gz"
        lmi_file = codecs.getreader('utf-8')(gzip.open(lmi_fpath), errors='replace') if is_gzip else codecs.open(
            lmi_fpath, "r", "utf-8")

        curr_word_dict = Counter()
        prev_word = ""
        for i, line in enumerate(lmi_file):
            if i % 1000000 == 0: print(i)
            try:
                word, feature, score = line.split("\t")
                score = float(score)
                curr_word = word

                if prev_word != curr_word and i > 0:
                    lmi_word2idx[prev_word] = len(lmi_word2idx)
                    lmi_list.append(curr_word_dict)
                    curr_word_dict = Counter()

                curr_word_dict[feature] += score
                prev_word = curr_word
            except:
                if self.VERBOSE: print(format_exc(), file=stderr)
        print()
        # add the last element
        lmi_word2idx[prev_word] = len(lmi_word2idx)
        lmi_list.append(curr_word_dict)

        if self.DEBUG:
            pprint(lmi_word2idx)
            for i, x in enumerate(lmi_list):
                print(i)
                print(x)
                print()

        dv = DictVectorizer(dtype=np.float32, separator='=', sparse=True)
        lmi_vectors = dv.fit_transform(lmi_list)

        vec_fpath = lmi_fpath + self.VECTORIZER_EXT
        joblib.dump(dv, vec_fpath)
        print("Vectorizer:", vec_fpath)

        print("Features Count:", len(dv.get_feature_names()))
        uniqf_fpath = lmi_fpath + self.FEATURES_EXT
        with codecs.open(uniqf_fpath, "w", "utf-8") as uniqf_file:
            for fn in dv.get_feature_names():
                print(fn, file=uniqf_file)
        print("Features:", uniqf_fpath)

        print("Word-feautre matrix shape:", lmi_vectors.shape)
        matrix_fpath = lmi_fpath + self.MATRIX_EXT
        joblib.dump(lmi_vectors, matrix_fpath)
        print("Word-feature matrix:", matrix_fpath)

        word2idx_fpath = lmi_fpath + self.WORD2IDX_EXT
        joblib.dump(lmi_word2idx, word2idx_fpath)
        print("Word2Index:", word2idx_fpath)
        lmi_file.close()

        self.word2idx = lmi_word2idx
        self.vectorizer = dv
        self.vectors = lmi_vectors

        return lmi_word2idx, dv.get_feature_names(), lmi_vectors

    @property
    def features(self):
        return self.vectorizer.get_feature_names()

    def similarity(self, word_i, word_j, unit_length=True):
        oov = word_i not in self.word2idx or word_j not in self.word2idx
        if oov:
            if self.VERBOSE: print("Warning: out of vocabulary:", word_i, word_j)
            return 0.0

        vector_i = self.vectors[self.word2idx[word_i]]
        vector_j = self.vectors[self.word2idx[word_j]]

        if unit_length:
            vector_i = vector_i / norm(vector_i)
            vector_j = vector_j / norm(vector_j)

        s_ij = vector_i.dot(vector_j.T)
        if s_ij != 0:
            return s_ij.data[0]
        else:
            return 0.0

    def max_similarity_pos(self, word_i, word_j, unit_length=True):
        words_i_pos = generate_mixed_cases(word_i, generate_pos=True)
        words_j_pos = generate_mixed_cases(word_j, generate_pos=True)

        sims_ij = []
        for w_i in words_i_pos:
            for w_j in words_j_pos:
                sims_ij.append((self.similarity(w_i, w_j, unit_length), w_i, w_j))

        sims_ij = sorted(sims_ij, reverse=True)
        if len(sims_ij) > 0:
            max_sim_ij = sims_ij[0][0]
            return max_sim_ij
        else:
            return 0.0
