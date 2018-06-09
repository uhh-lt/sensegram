import argparse
import codecs
import operator
from multiprocessing import Pool
from vector_representations.dense_sense_vectors import DenseSenseVectors
from traceback import format_exc


def generate_binary_hypers(dsv, inventory_fpath, max_synsets=1, hyper_synset_max_size=10, hc_max=0):
    output_fpath = inventory_fpath + ".vector-link-s%d-hmx%d-hc%d.csv" % (
        max_synsets, hyper_synset_max_size, hc_max)  
    bin_count = 0
    
    out = codecs.open(output_fpath, "w", "utf-8")
    log = codecs.open(output_fpath + ".log", "w", "utf-8")
    
    for i, h_id in enumerate(dsv.pcz.data):
        try:
            if i % 10000 == 0: print(i)

            if "h" in h_id:
                hypo_h_senses = dsv.pcz.data[h_id][0]["cluster"]
                tmp = sorted(dsv.pcz.data[h_id][0]["cluster"].items(), key=operator.itemgetter(1), reverse=True)

                s_id = "s" + h_id[1:]
                hypo_senses = dsv.pcz.data[s_id][0]["cluster"]
                log.write("\n{}\t{}\n".format(
                    h_id, ", ".join(hypo_h_senses)
                ))
                log.write("{}\n".format(
                    ", ".join(["{}:{}".format(k,v) for k,v in tmp])
                ))
                log.write("{}\t{}\n".format(
                    s_id, ", ".join(hypo_senses)
                ))

                # save relations from the hierarchical context 
                for hypo_sense in hypo_senses:
                    for hc_num, hyper_sense in enumerate(hypo_h_senses):
                        if hc_num == hc_max: break
                        hypo_word = hypo_sense.split("#")[0]
                        hyper_word = hyper_sense.split("#")[0]
                        if hypo_word != hyper_word:
                            out.write("{}\t{}\tfrom-original-labels\n".format(hypo_word, hyper_word))
                    bin_count += 1

                # save binary relations from a synset
                s_synsets = 0
                for rh_id, s in dsv.sense_vectors.most_similar(h_id + "#0"):
                    if "s" in rh_id:
                        hyper_senses = dsv.pcz.data[rh_id.split("#")[0]][0]["cluster"]
                        if len(hyper_senses) > hyper_synset_max_size: continue

                        rh_str = ", ".join(hyper_senses)
                        log.write("\t{}:{:.3f} {}\n".format(rh_id, s, rh_str))

                        for hypo_sense in hypo_senses:
                            for hyper_sense in hyper_senses:
                                hypo_word = hypo_sense.split("#")[0]
                                hyper_word = hyper_sense.split("#")[0]
                                if hypo_word != hyper_word:
                                    out.write("{}\t{}\tfrom-vector-linkage\n".format(hypo_word, hyper_word))
                                bin_count += 1
                        s_synsets += 1

                        if s_synsets >= max_synsets: break
        except KeyboardInterrupt:
            break
        except:
            print("Error", i, h_id)
            print(format_exc())
    out.close()
    log.close()
    
    print("# binary relations:", bin_count)
    print("binary relations:", output_fpath)
    print("log of binary relations:", output_fpath + ".log")
    
    return bin_count, output_fpath

    
def run(inventory_fpath):
    dsv = DenseSenseVectors(
        pcz_fpath=inventory_fpath,
        word_vectors_obj=None,
        save_pkl=True,
        sense_dim_num=1000,
        norm_type="sum",
        weight_type="score",
        max_cluster_words=20)

    for max_top_synsets in [1]:
        for max_hyper_synset_size in [3, 5, 10, 15, 20]:
            for hc_max in [1, 2, 3]: 
                print("="*50)
                print("max number of synsets:", max_top_synsets)
                print("max hyper synset size:", max_hyper_synset_size)
                print("hc_max:", hc_max)
                generate_binary_hypers(dsv, inventory_fpath, max_top_synsets, max_hyper_synset_size, hc_max)
                       
  
def main():
    parser = argparse.ArgumentParser(description='Generate binary hypernymy relations via linking of hyper synsets to regular synsets.' + 
        'The program saves the output files next to the input file (multiple files with multiple parameters are generated).')
    parser.add_argument('inventory', help='Path to an input PCZ file with synsets clusters in the format: "word<TAB>cid<TAB>clusters<TAB>",' + 
        ' where <cluster> is "word:sim, word:sim, ...", e.g. "rwn-patterns-limit-tfidf-sensegram.tsv".' +
        ' The file should contain "h" synsets and "s" synsets starting from the respective' +
        ' letters, e.g. s1234 and h1234, to work properly for this method. The program expects that, .sense_vectors file is located '+ 
        ' next to the pcz file and will not function properly without this file.')
    args = parser.parse_args()
    print("Input inventory of synsets and hyper synsets:", args.inventory)
    run(args.inventory)
    

if __name__ == '__main__':
    main()
