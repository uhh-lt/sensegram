#!/usr/bin/env python
# encoding: utf-8

from os.path import splitext, join
import codecs
import traceback
import argparse
from shutil import copyfile
import fileinput
from pandas import read_csv
import os
from collections import defaultdict 
from numpy import std, mean, median


HEADER_IN = "word\tcid\tkeyword\tcluster"
HEADER_OUT = "word\tcid\tcluster\tisas"
CHUNK_LINES = 500000


def add_header(input_fpath, header):
    for line in fileinput.input(files=[input_fpath], inplace=True):
        if fileinput.isfirstline():
            print header
        print line,


def try_remove(fpath):
    if exists(fpath):
        os.remove(fpath)


def exists(dir_path):
    return os.path.isdir(dir_path) or os.path.isfile(dir_path)


def postprocess(ddt_fpath, output_fpath, filtered_fpath, min_size):
    print "Input DDT:", ddt_fpath
    print "Output DDT:", output_fpath
    print "Filtered out DDT clusters:", filtered_fpath
    print "Min size:", min_size

    min_size = int(min_size)
    ddt_tmp_fpath = ddt_fpath + ".tmp"
    copyfile(ddt_fpath, ddt_tmp_fpath)
    add_header(ddt_tmp_fpath, HEADER_IN)
    
    with codecs.open(output_fpath, "w", encoding="utf-8") as output, codecs.open(filtered_fpath, "w", encoding="utf-8") as filtered:
        reader = read_csv(ddt_tmp_fpath, encoding="utf-8", delimiter="\t", error_bad_lines=False,
            iterator=True, chunksize=CHUNK_LINES, doublequote=False, quotechar=u"\u0000")
        num = 0
        selected_num = 0
        senses_num = defaultdict(int)

        for i, chunk in enumerate(reader):
            # print header
            if i == 0: output.write(HEADER_OUT + "\n")
            chunk.fillna('')
            
            # rows
            for j, row in chunk.iterrows():
                num += 1
                
                # filters
                cluster = row.cluster.split("  ")
                if len(cluster) < min_size:
                    filtered.write("%s\t%d\t%s\n" % (row.word, row.cid, ",".join(cluster)))
                    continue
                output.write("%s\t%d\t%s\t\n" % (row.word, row.cid, ",".join(cluster)))
                selected_num += 1
                senses_num[row.word] += 1

        print "# output clusters: %d of %d (%.2f %%)" % (selected_num, num, float(selected_num)/num*100.)
        values = senses_num.values()
        print "average number of senses: %.2f +- %.3f, median: %.3f" % (mean(values), std(values), median(values))
    try_remove(ddt_tmp_fpath)


def main():
    parser = argparse.ArgumentParser(description='Postprocess word sense induction file.')
    parser.add_argument('ddt', help='Path to a csv file with a DDT: "word<TAB>sense-id<TAB>keyword<TAB>cluster"  w/o header by default. Here <cluster> is "word:sim<SPACE><SPACE>word:sim<SPACE><SPACE>..."')
    parser.add_argument('-min_size', help='Minimum cluster size. Default -- 5.', default="5")
    args = parser.parse_args()

    output_fpath = splitext(args.ddt)[0] + "-minsize" + args.min_size + ".csv"
    filtered_fpath = splitext(args.ddt)[0] + "-minsize" + args.min_size + "-filtered.csv"

    postprocess(args.ddt, output_fpath, filtered_fpath, args.min_size)

if __name__ == '__main__':
    main()
