#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filter clusters produced by chinese-whispers. Delete clusters that are too small.
This program was adapted from postprocess.py script distributed with chinese-whispers implementation.
"""

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

def build_output_fpath(clusters_fpath, minsize):
        return splitext(clusters_fpath)[0] + "_minsize" + minsize + ".csv"

def build_filtered_fpath(clusters_fpath, minsize):
        return splitext(clusters_fpath)[0] + "_minsize" + minsize + "_filtered.csv"


def run(ddt_fpath, output_fpath=None, filtered_fpath=None, min_size="5"):
    """ 
    This function filters clusters produced by chinese-whispers clustering algorithm.
    It deletes clusters that are smaller than min_size.

    Args:
        ddt_fpath:  a path to an input file with clusters.
                    (output of chinese-whispers algorithm).
                    Format: word<TAB>sense_id<TAB>keyword<TAB>cluster where cluster is word:sim<SPACE><SPACE>word:sim<SPACE><SPACE>...
                    The file is without header.
        output_fpath:   a path to an output file with clusters.
                        Changed format:
                        word<TAB>sense_id<TAB>cluster<TAB>isas where cluster is word:sim,word:sim...
                        The file has a header.
        filtered_fpath: a path to a file that holds all filtered clusters. Same format as in output file.
        min_size:   a minimal accepted size of a cluster. By default = "5". Arg type is string.

    Returns:
        selected_num:   number of clusters in the output_fpath file
        mean(senses):   average number of senses per word
    """

    output_fpath = output_fpath or build_output_fpath(ddt_fpath, min_size) 
    filtered_fpath = filtered_fpath or build_filtered_fpath(ddt_fpath, min_size)

    print "Input:", ddt_fpath
    print "Output:", output_fpath
    print "Filtered out clusters:", filtered_fpath
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
            chunk = chunk.fillna('')
            
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

        print "output clusters: %d of %d (%.2f %%)" % (selected_num, num, float(selected_num)/num*100.)
        values = senses_num.values()
        print "average number of senses: %.2f +- %.3f, median: %.3f" % (mean(values), std(values), median(values))
    try_remove(ddt_tmp_fpath)
    return selected_num, mean(values)

    
def main():
    parser = argparse.ArgumentParser(description='Postprocess sense clusters. Delete clusters smaller than minsize.')
    parser.add_argument('ddt', help='Path to an input file with clusters. DDT format: "word<TAB>sense-id<TAB>keyword<TAB>cluster"  w/o header by default. Here <cluster> is "word:sim<SPACE><SPACE>word:sim<SPACE><SPACE>..."')
    parser.add_argument('-min_size', help='Minimum cluster size. Default -- 5.', default="5")
    args = parser.parse_args()
    
    output_fpath = build_output_fpath(args.ddt, args.min_size) 
    filtered_fpath = build_filtered_fpath(args.ddt, args.min_size)

    run(args.ddt, output_fpath, filtered_fpath, args.min_size)
    

if __name__ == '__main__':
    main()
