import codecs
import argparse
from pandas import read_csv
from collections import defaultdict
from numpy import std, mean, median


HEADER = "word\tcid\tcluster\tisas"
CHUNK_LINES = 500000


def run(ddt_fpath, output_fpath, min_size=5):
    with codecs.open(output_fpath, "w", encoding="utf-8") as output:
        reader = read_csv(ddt_fpath, encoding="utf-8", delimiter="\t", error_bad_lines=False,
            iterator=True, chunksize=CHUNK_LINES, doublequote=False, quotechar="\u0000")
        num = 0
        selected_num = 0
        senses_num = defaultdict(int)

        for i, chunk in enumerate(reader):
            if i == 0: output.write(HEADER + "\n")
            chunk = chunk.fillna('')
            
            for j, row in chunk.iterrows():
                num += 1
                
                cluster = row.cluster.split(",")
                if len(cluster) < min_size: continue
                output.write("{}\t{:d}\t{}\t\n".format(row.word, row.cid, row.cluster))
                selected_num += 1
                senses_num[row.word] += 1

        print("Output senses: %d of %d (%.2f %%)" % (selected_num, num, float(selected_num)/num*100.))
        values = list(senses_num.values())
        print("Average number of senses: %.2f +- %.3f, median: %.3f" % (mean(values), std(values), median(values)))
    
    return selected_num, mean(values)

    
def main():
    parser = argparse.ArgumentParser(description='Postprocess sense clusters. Delete clusters smaller than min. size.')
    parser.add_argument('ddt', help='Path to an input file with clusters in the format: "word<TAB>cid<TAB>clusters<TAB>", where <cluster> is "word:sim, word:sim, ..."')
    parser.add_argument('-min_size', help='Minimum cluster size. Default -- 5.', type=int, default=5)
    args = parser.parse_args()
    output_fpath = args.ddt + ".output"

    print("Input:", args.ddt)
    print("Output:", output_fpath)
    print("Min size:", args.min_size)

    run(args.ddt, output_fpath, args.min_size)
    

if __name__ == '__main__':
    main()
