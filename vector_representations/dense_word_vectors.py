import gensim

class DenseWordVectors:
    def __init__(self, w2v_fpath):
        self.vectors = gensim.models.KeyedVectors.load_word2vec_format(
            w2v_fpath,
            binary=False,
            unicode_errors='ignore')

        self.vectors.init_sims(replace=True)

