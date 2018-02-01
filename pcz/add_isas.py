from pandas import read_csv
import codecs
from .isas import ISAs
from collections import Counter
import argparse
from utils.morph import lemmatize_word
import spacy


LIST_SEP = ","
DDT_HEADER = "word\tcid\tcluster\tisas"
MAX_HYPERS = 5
SENSE_SEP = "#"
SCORE_SEP = ":"
VERBOSE = True
MIN_SUBSTRINGABLE_LENGTH = 6
ADPOSITION = "adp"
NOUN = "noun"


_spacy = spacy.load('en')


def read_ddt(ddt_fpath):
    df = read_csv(ddt_fpath, "\t", encoding='utf8', error_bad_lines=False, doublequote=False, quotechar="\u0000")
    df.word.fillna("", inplace=True)
    df.cid.fillna(-1, inplace=True)
    df.cluster.fillna("", inplace=True)
    df.isas.fillna("", inplace=True)
    return df


def analyze(term):
    tokens = _spacy(term, tag=True, parse=False, entity=True)
    tok_pos_ent = []

    for t in tokens:
        tok_pos_ent.append((t.orth_, t.pos_, "I" if t.ent_iob_.lower() != "o" else "O"))

    return tok_pos_ent


def substring_hyper(term):
    """ Returns hypernym based on the substrings if any, else return just void string.
    Along with the hypernym, returns also score 0-1. """

    if len(term) <= MIN_SUBSTRINGABLE_LENGTH or " " not in term: return "", 0.0

    tok_pos_ent = analyze(term)
    adp_index = -1
    has_entity = False
    for i, tpe in enumerate(tok_pos_ent):
        if tpe[1].lower() == ADPOSITION: adp_index = i
        if tpe[2].lower() == "i": has_entity = True

    if has_entity: return "", 0.0

    if adp_index != -1:
        hyper = " ".join([t for t,p,e in tok_pos_ent[0:adp_index]])
        hyper_head_pos = tok_pos_ent[adp_index - 1][1]
    else:
        hyper = tok_pos_ent[-1][0]
        hyper_head_pos = tok_pos_ent[-1][1]

    if hyper_head_pos.lower() != NOUN: return "", 0.0
    else: return hyper, float(len(hyper)) / float(len(term))


def add_isas(ddt_fpath, output_fpath, isas_fpath, max_hypers=MAX_HYPERS):
    hypers = ISAs(isas_fpath, min_freq=0.0, preprocess=True, sep='\t', strip_pos=True, use_pickle=True, lowercase=True)
    cluster_err_count = 0

    with codecs.open(output_fpath, "w", "utf-8") as output:
        df = read_ddt(ddt_fpath)
        print(DDT_HEADER, file=output)

        for i, row in df.iterrows():
            cluster = row.cluster.split(LIST_SEP)
            cluster_hypers = Counter()
            cluster_hypers_count = Counter()

            for cw in cluster:
                try:
                    word_sense, score = cw.split(SCORE_SEP)
                    if SENSE_SEP in word_sense: word = word_sense.split(SENSE_SEP)[0]
                    else: word = word_sense
                    word_hypers = hypers.all_hyper(word)
                    for hyper, freq in word_hypers:
                        cluster_hypers[hyper] += float(score) * freq
                        cluster_hypers_count[hyper] += 1
                except:
                    if VERBOSE and cluster_err_count < 1000:
                        print("Warning: wrong cluster word:", cw)
                    cluster_err_count += 1

            cluster_hypers.pop(row.word, None)
            cluster_hypers.pop(row.word.upper(), None)
            cluster_hypers_count.pop(row.word, None)
            cluster_hypers_count.pop(row.word.upper(), None)
            if len(row.word) > 2: cluster_hypers.pop(row.word[0].upper() + row.word[1].lower(), None)
            clusters_hyper_top = cluster_hypers.most_common(max_hypers)
            isas_lst = [(hyper, hyper_score) for hyper, hyper_score in clusters_hyper_top if lemmatize_word(hyper) != row.word.lower()]
            isas = [hyper + SCORE_SEP + str(hyper_score) for hyper, hyper_score in isas_lst]
            substr_hyper, substr_score = substring_hyper(row.word)
            if substr_hyper != "" and substr_hyper not in clusters_hyper_top: isas.append(substr_hyper + SCORE_SEP + str(substr_score))
            
            isas_coverage = []
            for hyper, score in isas_lst:
                isas_coverage.append("%s:%.3f:%.3f:%d:%.3f" % (hyper, cluster_hypers[hyper], cluster_hypers[hyper]/float(len(cluster)),
                    cluster_hypers_count[hyper], cluster_hypers_count[hyper]/len(cluster)))

            print("%s\t%d\t%s\t%s\t%s" % (row.word, row.cid, LIST_SEP.join(cluster), 
                    LIST_SEP.join(isas), LIST_SEP.join(isas_coverage)), file=output)

    print("# cluster errors:", cluster_err_count)
    print("Output:", output_fpath)


def main():
    parser = argparse.ArgumentParser(description='Add ISA relations to sense clusters or DDT.')
    parser.add_argument('ddt', help='Path to a csv file with disambiguated or unsiambiguated sense clusters'
                                    ' "word<TAB>cid<TAB>cluster<TAB>isas", where isas are void and cluster element can'
                                    'optionally have sense id e.g. "python#9:0.99".')
    parser.add_argument('isas', help='ISAs file. Path to a CSV file "hypo<TAB>hyper<TAB>freq". If a pre-calculated'
                                     '.pkl file is found next to this file it will be loaded for speed.')
    parser.add_argument('--output', help='Output path. Default -- next to input file.', default="")
    parser.add_argument('--max_hyper_num', help='Maximum number of pattern-based hypernyms (total size is +1 at most based on '
                                   'substring hypernyms where available). Default -- %d.' % MAX_HYPERS, default=MAX_HYPERS)
    args = parser.parse_args()

    output_fpath = args.ddt + ".isas" if args.output == "" else args.output
    print("Input sense clusters: ", args.ddt)
    print("Output path: ", output_fpath)
    print("Max. number of hypernyms: ", args.max_hyper_num)
    add_isas(ddt_fpath=args.ddt, output_fpath=output_fpath, isas_fpath=args.isas, max_hypers=int(args.max_hyper_num))


if __name__ == '__main__':
    main()
