import argparse
from vector_representations.dense_sense_vectors import DenseSenseVectors
from vector_representations.dense_word_vectors import DenseWordVectors
from vector_representations.sparse_word_vectors import SparseWordVectors
from vector_representations.sparse_sense_vectors import SparseSenseVectors
from os.path import exists


def run(pcz_fpath, wv_fpath, sparse=False, sense_dim_num=1000, save_pkl=False, norm_type="sum", weight_type="score", max_cluster_words=20):
    print("Input PCZ:", pcz_fpath)
    print("Input word vectors:", wv_fpath)
    print("Sparse:", sparse)
    print("Type of vector normalization:", norm_type)
    print("Weight type:", weight_type)
    print("Max. number of cluster words to use:", max_cluster_words)
    print("Sense dim. number (sparse only):", sense_dim_num)
    print("Save pickle (sparse only):", save_pkl)

    if exists(pcz_fpath) and exists(wv_fpath):
        if sparse:
            WV = SparseWordVectors
            SV = SparseSenseVectors
        else:
            WV = DenseWordVectors
            SV = DenseSenseVectors

        wv = WV(wv_fpath)
        sv = SV(
            pcz_fpath,
            wv,
            save_pkl=save_pkl,
            sense_dim_num=sense_dim_num,
            norm_type=norm_type,
            weight_type=weight_type,
            max_cluster_words=max_cluster_words)
    else:
        print("Input paths not found.")
        print(exists(pcz_fpath), pcz_fpath)
        print(exists(wv_fpath), wv_fpath)


def main():
    parser = argparse.ArgumentParser(description="Build sense vectors out of sense inventory and word vectors.")
    parser.add_argument('pcz', help='PCZ in the format "word<TAB>cid<TAB>cluster<TAB>isas". Cluster and isas'
                                    ' are comma separated lists of "word#cid:score" or "word#pos#cid:score items". ')
    parser.add_argument('wv', help='Word vectors file in the CSV format "word<TAB>feature<TAB>score" strictly sorted'
                                   ' by "word,score" if sparse. Otherwise dense word vectors in the word2vec format.'
                                   'The file should contain exactly three columns and have no header. ')
    parser.add_argument('-d', '--max_dim', type=int, default=1000, help='Max. number of non zero elements in a ' \
                        'sense vector. Default: 1000.')
    parser.add_argument('-m', '--max_words', type=int, default=20, help='Max. number of cluster words to use for'
                                                                                  ' averaging. Default: 20.')
    parser.add_argument('-n', '--norm_type', choices=['sum', 'no'], default="sum", help='Type of normalization '
                        'of the output vectors: sum (divide by sum of scores), no (no normalization). Default: sum.')
    parser.add_argument('-p', '--no_pkl', action='store_true', help='Save no binary sense vectors (smaller memory foodprint).')
    parser.add_argument('-s', '--sparse', action='store_true', help='Input word vectors are sparse vectors (e.g. LMI),'
                                                              ' otherwise they are considered dense word2vec vectors.')
    parser.add_argument('-w', '--weight_type', choices=['score', 'rank', 'ones'], help='Type of weighting of words in a cluster.'
                                              ' Options: score (use similarity score of target word with the cluster word), rank '
                                              '(use inversed rank of the position of the word in the cluster as sorted by the similarity scores),'
                                              ' ones (each word in the cluseter contributes equally i.e. no scores). '
                                              'Default: score.', default="score")
    args = parser.parse_args()
    run(args.pcz, args.wv, args.sparse, int(args.max_dim), (not args.no_pkl),
        args.norm_type, args.weight_type, args.max_words)


if __name__ == '__main__':
    main()

