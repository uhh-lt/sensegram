# coding=utf8

import argparse
import numpy as np
import math
import random
from sys import stderr, stdout
import six
import pprint
from os.path import splitext, join
import codecs
from .patterns import re_escape, re_amp, re_quote_escape
from pandas import read_csv
from itertools import islice
import pickle as pickle
import os
from os.path import join, abspath, dirname
import gzip
from ntpath import basename

""" This namespace contains a set of small common purpose functions and constants. """

UNK_LABEL = "unknown"
TRUE = ['true', '1', 't', 'y', 'yes']
LETTERS = ['а','б','в','г','д','е','ё','ж','з','и','к','л','м','н','о','п','р','с','т','у','ф','х','ц','ч','ш','щ','э','ь','ы','ю','я','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


def fpath2filename(fpath):
    """ Returns filename without up to two extensions e.g. /Users/alex/work/joint/src/isas-cc.csv.gz --> isas-cc """
    return splitext(splitext(basename(fpath))[0])[0]


def get_data_dir():
    return abspath(join(join(dirname(module.__file__), os.pardir), "data"))


def dt_scientific2fixed(dt_fpath, output_fpath):
    """ Convert similarity from scientific to normal format. """

    dt = read_csv(dt_fpath, "\t", encoding='utf8', error_bad_lines=False)
    dt = dt.sort(["sim"], ascending=[0])
    dt.to_csv(output_fpath, sep="\t", encoding="utf-8", float_format='%.12f', index=False)


def strip_header(input_fpath):
    import fileinput

    for line in fileinput.input(files=[input_fpath], inplace=True):
        if fileinput.isfirstline():
            continue
        print(line, end="")


def add_header(input_fpath, header):
    import fileinput

    for line in fileinput.input(files=[input_fpath], inplace=True):
        if fileinput.isfirstline():
            print(header)
        print(line, end="")


def base(fpath):
    return base_ext(fpath)[0]


def base_ext(fpath):
    components = splitext(fpath)  
    if len(components) < 2:
        return components[0], ""
    else:
        return components[0], components[1]


def prt(string):
    stdout.write("%s\n" % string)


def prt2(tuple2):
    stdout.write("%s %s\n" % (tuple2[0], tuple2[1]))


def profiling(function):
    import cProfile
    import pstats
    from io import StringIO
    pr = cProfile.Profile()
    pr.enable()

    function()

    pr.disable()
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    stderr.write(s.getvalue())


def ensure_utf8(text):
    """ Make sure that the string is in unicode. """

    if six.PY2 and not isinstance(text, six.text_type): 
        return text.decode('utf-8')
    else: return text


def list2str(lst, short=True):
    """ Returns a string representing a list """

    try:
        if short:
            return ', '.join(lst)
        else:
            return str(lst)
    except:
        if short:
            return ""
        else:
            return "[]"


def str2list(str_list):
    """ Parses a string that supposed to contain a list
        (or something that has len). Returns a list. """

    try:
        l = eval(str_list)
        if hasattr(l, "__len__"):
            return l
        else:
            print("Warning: cannot parse '%s'. " % str_list, file=stderr)
            return []
    except:
        print("Warning: cannot parse '%s'. " % str_list, file=stderr)
        return []


def random_ints():
    """ Returns a random integer from 0 to 100,000 """
    return str(int(math.floor(random.random() * 100000)))


from .patterns import re_newlines
def strip_newlines(input):
    return re_newlines.sub(" ", input)


from .patterns import re_whitespaces
def normalize_whitespaces(input):
    return re_whitespaces.sub(" ", input)

from .patterns import re_url
def get_urls(input):
    matches = re_url.findall(input)
    return matches


def findnth(haystack, needle, n):
    parts = haystack.split(needle, n + 1)
    if len(parts) <= n + 1:
        return -1
    return len(haystack) - len(parts[-1]) - len(needle)


def whatisthis(s):
    if isinstance(s, str):
        return "str"
    elif isinstance(s, str):
        return "unicode"
    else:
        return "not str"


def exists(dir_path):
    return os.path.isdir(dir_path) or os.path.isfile(dir_path)


def try_remove(fpath):
    if exists(fpath):
        os.remove(fpath)


def safe_remove(fpath):
    try:
        os.remove(fpath)
        print("File removed:", fpath)
    except OSError:
        print("Cannot remove file:", fpath)

def ensure_dir(f):
    """ Make the directory. """
    if not os.path.exists(f): os.makedirs(f)


def chunks(l, n):
    """ Yield successive n-sized chunks from l. """

    for i in range(0, len(l), n):
        yield list(zip(list(range(i,i+n)), l[i:i+n]))


def stat(lst, print_stat=True):
    sizes_arr = np.array(lst)
    s = {}
    s["sum"] = np.sum(sizes_arr, axis=0)
    s["mean"] = np.mean(sizes_arr, axis=0)
    s["std"] = np.std(sizes_arr, axis=0)
    s["median"] = np.median(sizes_arr, axis=0)
    s["min"] = np.min(sizes_arr, axis=0)
    s["max"] = np.max(sizes_arr, axis=0)
    if print_stat:
        print("number:", s["sum"], file=stderr)
        print("mean: %.0f +- %.0f" % (s["mean"], s["std"]), file=stderr)
        print("median: %.0f" % s["median"], file=stderr)
        print("min: %.0f" % s["min"], file=stderr)
        print("max: %.0f" % s["max"], file=stderr)
    return s


class readable_dir(argparse.Action):
    """ Required for argparse parse.add_argument method. """

    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))


class PrettyPrinterUtf8(pprint.PrettyPrinter):
    def format(self, object, context, maxlevels, level):
        if isinstance(object, str):
            return (object.encode('utf8'), True, False)
        return pprint.PrettyPrinter.format(self, object, context, maxlevels, level)


def print_line():
    print("...............................................................")


def load_voc(voc_fpath, preprocess=True, sep='\t', use_pickle=True, silent=False):
    """ Reads vocabulary in the "word" format """

    pkl_fpath = voc_fpath + ".pkl"
    if use_pickle and exists(pkl_fpath):
        voc = pickle.load(open(pkl_fpath, "rb"))
    else:
        if preprocess:
            freq_cln_fpath = voc_fpath + "-cln"
            preprocess_pandas_csv(voc_fpath, freq_cln_fpath)
        else:
            freq_cln_fpath = voc_fpath

        word_df = read_csv(freq_cln_fpath, sep, encoding='utf8', error_bad_lines=False)
        voc = set(row["word"] for i, row in word_df.iterrows())
        print("vocabulary is loaded:", len(voc))

        if use_pickle:
            pickle.dump(voc, open(pkl_fpath, "wb"))
            print("Pickled voc dictionary:", pkl_fpath)

    if not silent: print("Loaded %d words from: %s" % (len(voc), pkl_fpath if pkl_fpath else voc_fpath))

    return voc


def gunzip_file(input_gzipped_fpath, output_gunzipped_fpath):
    with codecs.open(output_gunzipped_fpath, "wb") as out:
        input_file = gzip.open(input_gzipped_fpath, "rb")
        try:
            out.write(input_file.read())
        finally:
            input_file.close()


def preprocess_pandas_csv(input_fpath, output_fpath=""):

    prefix, ext = splitext(input_fpath)
    if ext == ".gz":
        gunzipped_input_fpath = prefix + ".csv"
        gunzip_file(input_fpath, gunzipped_input_fpath)
        input_fpath = gunzipped_input_fpath

    out_fpath = output_fpath if output_fpath != "" else input_fpath + ".tmp"
    with codecs.open(input_fpath, "r", "utf-8") as input, codecs.open(out_fpath, "w", "utf-8") as output:
        for line in input:
            s = line.strip()
            s = re_amp.sub(" ", s)
            s = re_escape.sub(" ", s)
            s = re_quote_escape.sub(" ", s)
            print(s, file=output)

    if output_fpath == "":
        try_remove(input_fpath)
        os.rename(out_fpath, input_fpath)
        print("cleaned csv:", input_fpath)
    else:
        print("cleaned csv:", output_fpath)

    if ext == ".gz": try_remove(gunzipped_input_fpath)

def lowercase_voc(voc):
    """ In case of conflict take the max of two. """

    print("....")

    vocl = {}

    for v in voc:
        vl = v.lower()

        if vl not in vocl or vocl[vl] < voc[v]:
            vocl[vl] = voc[v]
        else:
            pass

    return vocl


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))
