import codecs
from pandas import read_csv 
from .dense_sense_vectors import DenseSenseVectors
from .dense_word_vectors import DenseWordVectors
from time import time 
from os.path import exists

def evalutate(sv, test_fpath, output_fpath):
    with codecs.open(output_fpath, "w", "utf-8") as out:
        print("word1,word2,sim", file=out)

        test = read_csv(test_fpath, sep=",", encoding="utf-8")
        for i, row in test.iterrows():
            sim12 = sv.max_pairwise_sim(row.word1, row.word2)
            print("%s,%s,%.6f" % (row.word1, row.word2, sim12), file=out)
            
    print("Output:", output_fpath)
    
def evalutate_original(w2v, test_fpath, output_fpath):
    with codecs.open(output_fpath, "w", "utf-8") as out:
        print("word1,word2,sim", file=out)

        skipped = 0
        test = read_csv(test_fpath, sep=",", encoding="utf-8")
        for i, row in test.iterrows():
            if row.word1 in w2v and row.word2 in w2v:
                sim12 = w2v.similarity(row.word1, row.word2)
            else:
                print("Skipping: '%s', '%s'" % (row.word1, row.word2))
                sim12 = 0.0
                skipped += 1
            print("%s,%s,%.6f" % (row.word1, row.word2, sim12), file=out)
    print("Output:", output_fpath)
    print("Skipped:", skipped)

    
# Evaluating new models on the RUSSE testset

test_fpath = "/mnt1/verbs/russe-evaluation/russe/evaluation/test.csv"
recalculate = False

for weight_type in ["rank", "ones"]:
    for max_cluster_words in [999]: #, 5, 10, 20]:
        for model in ["w2v", "jbt"]:
            print("="*50)
            print("Loading", max_cluster_words, model, weight_type)
            output_fpath = test_fpath + "." + model + "-" + str(max_cluster_words) + "-" + weight_type
            if exists(output_fpath):
                print("Results exist, skipping:", output_fpath)
                continue
            
            tic = time()
            if recalculate:
                wv = DenseWordVectors("/mnt1/verbs/all.norm-sz500-w10-cb0-it1-min100.w2v")
            else:
                wv = None
                
            dsv = DenseSenseVectors(pcz_fpath="/mnt1/verbs/%s.csv.gz" % (model),
                                    word_vectors_obj=wv,
                                    save_pkl=True,
                                    sense_dim_num=1000,
                                    norm_type="sum",
                                    weight_type=weight_type,
                                    max_cluster_words=max_cluster_words)
            
            evalutate(dsv, test_fpath, output_fpath)
            print(time() - tic, "sec.")

