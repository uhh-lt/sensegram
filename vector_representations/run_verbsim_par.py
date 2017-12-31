import argparse
from pandas import read_csv
from multiprocessing import Pool
from contextlib import contextmanager
from time import time
from .sparse_word_vectors import SparseWordVectors
from .sparse_sense_vectors import SparseSenseVectors


ssv = None


@contextmanager
def terminating(thing):
    try: yield thing
    finally: thing.terminate()
        
def calculate_corr(p):
    simlex_fpath, gold_col, use_word_vectors = p[0], p[1], p[2]
    simlex = read_csv(simlex_fpath, sep="\t", encoding="utf-8")
    simlex["result"] = 0.0
    for i, row in simlex.iterrows():
        if i % 100 == 0: print(i)
            
        simlex.loc[i,"result"] = ssv.max_similarity_pos(
            row.word1, row.word2, unit_norm=True, use_word_vectors=use_word_vectors)
    scorr = simlex[gold_col].corr(simlex["result"], method="spearman")
    return (scorr, simlex_fpath, use_word_vectors)
        
    
def run(pcz_fpath, lmi_fpath, todo, num_cores):
    global ssv
    
    tic = time()
    swv = SparseWordVectors(lmi_fpath)
    ssv = SparseSenseVectors(
        pcz_fpath,
        swv,
        sense_dim_num=1000,
        max_cluster_words=20)


    if num_cores > 1:
        with terminating(Pool(num_cores)) as pool:
            for res in pool.imap_unordered(calculate_corr, todo):
                print(res)
    else:
        for p in todo:
            res = calculate_corr(p)
            print(res)

    print((time()-tic))


def main():
    parser = argparse.ArgumentParser(description="Build sense vectors out of sense inventory and word vectors.")
    parser.add_argument('pcz', help='PCZ in the format "word<TAB>cid<TAB>cluster<TAB>isas"')
    parser.add_argument('--nopar', action='store_true', help='Use no parallelization. Default -- false.')
    args = parser.parse_args()
    
    todo = [
        ("/mnt10/verbs/sim/SimLex-999/SimLex-999.txt", "SimLex999", True),
        ("/mnt10/verbs/sim/data/SimVerb-3500.txt", "score", True),
        ("/mnt10/verbs/sim/data/SimVerb-3000-test.txt", "score", True),
        ("/mnt10/verbs/sim/data/SimVerb-500-dev.txt", "score", True),
        ("/mnt10/verbs/sim/SimLex-999/SimLex-999.txt", "SimLex999", False),
        ("/mnt10/verbs/sim/data/SimVerb-3500.txt", "score", False),
        ("/mnt10/verbs/sim/data/SimVerb-3000-test.txt", "score", False),
        ("/mnt10/verbs/sim/data/SimVerb-500-dev.txt", "score", False)]
    lmi_fpath = "/mnt10/verbs/data/culwg/release/lmi-culwg-coarse.csv.gz"

    if args.nopar: num_cores = 1
    else: num_cores = 8
    run(args.pcz, lmi_fpath, todo, num_cores)


if __name__ == '__main__':
    main()

