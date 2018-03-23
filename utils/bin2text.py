import gensim

binary_w2v_fpath = "model/ukwac.jbt.sense_vectors"
text_w2v_fpath = "model/ukwac.jbt.txt.sense_vectors"

vectors = gensim.models.KeyedVectors.load_word2vec_format(
            binary_w2v_fpath,
            binary=True,
            unicode_errors='ignore')

vectors.init_sims(replace=True)
vectors.save_word2vec_format(text_w2v_fpath)
