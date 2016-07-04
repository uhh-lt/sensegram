#!/usr/bin/env python
# encoding: utf-8

import argparse
from os.path import splitext, join
import os
from postprocess import postprocess
import glob 

def main():
    parser = argparse.ArgumentParser(description='Postprocess word sense induction file for all files in a directory.')
    parser.add_argument('ddt_dir', help='Path to a directory with csv files with DDTs: "word<TAB>sense-id<TAB>keyword<TAB>cluster"  w/o header by default. Here <cluster> is "word:sim<SPACE><SPACE>word:sim<SPACE><SPACE>..."')
    parser.add_argument('-min_size', help='Minimum cluster size. Default -- 5.', default="5")
    args = parser.parse_args()

    print "Input DDT directory (pattern):", args.ddt_dir
    print "Min size:", args.min_size

    #postprocess(args.ddt, output_fpath, filtered_fpath, int(args.min_size))
    for cluster_fpath in glob.glob(args.ddt_dir):
        if splitext(cluster_fpath)[-1] == ".csv":
            print "\n>>>", cluster_fpath
            postprocess(
                    cluster_fpath,
                    cluster_fpath+"-minsize" + args.min_size + ".csv",
                    cluster_fpath+"-minsize" + args.min_size + "-filtered.csv",
                    args.min_size)

if __name__ == '__main__':
    main()
