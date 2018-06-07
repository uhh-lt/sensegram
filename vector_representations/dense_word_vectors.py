from gensim.models import KeyedVectors
from os.path import exists
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class DenseWordVectors:
    def __init__(self, w2v_fpath):
        pkl_fpath = w2v_fpath + ".pkl"

        if exists(pkl_fpath):
            self.vectors = KeyedVectors.load(pkl_fpath)
        else:
            self.vectors = KeyedVectors.load_word2vec_format(
                w2v_fpath,
                binary=False,
                unicode_errors='ignore')

        self.vectors.init_sims(replace=True)

