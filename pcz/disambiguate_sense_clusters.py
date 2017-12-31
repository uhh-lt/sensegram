import sys
from time import time
from .sense_clusters import SenseClusters
from utils.morph import lemmatize_word, analyze_word
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer
from numpy import argmax
import codecs
from operator import itemgetter
import argparse
from traceback import format_exc
from collections import Counter
from utils.common import load_voc, exists


SEP = "\t"
SEP3 = ":"
SEP4 = "#"
NORM = "l2"
MIN_SCORE = 0.000001
OUTPUT_UNKNOWN = True


class SenseClusterDisambiguator(object):
    def __init__(self, sense_clusters_fpath, skip_voc_fpath=""):
        self._skip_voc = load_voc(skip_voc_fpath, preprocess=True, sep='\t', use_pickle=True) if exists(skip_voc_fpath) else set()
        print("Skip voc:", len(self._skip_voc))
        self._sense_clusters = SenseClusters(sense_clusters_fpath, strip_dst_senses=False)
        self._sc = self._sense_clusters.data

    def run(self, output_fpath, normalize=True, output_sim=False, skip_ambigous=False):

        with codecs.open(output_fpath, "w", "utf-8") as output_file:
            print("word\tcid\tcluster\tisas", file=output_file)
            err_num = 0
            i = 0

            # foreach word
            for i, word in enumerate(self._sc.keys()):
                if word in self._skip_voc:
                    print(word, end=' ')
                    continue

                if word not in self._sc:
                    print("Warning: skipping word", word)
                    err_num += 1
                    continue
                if i % 5000 == 0: print(i, "sense clusters processed")

                # foreach word sense
                for sense_id in self._sc[word]:
                    ddt_cluster = self._disambiguate(target_field="cluster", context_fields=["cluster"], sense_fields=["cluster"],
                        normalize=normalize, sense_id=sense_id, word=word, skip_ambigous=skip_ambigous)
                    ddt_isas = self._disambiguate(target_field="isas", context_fields=["cluster","isas"], sense_fields=["cluster"],
                        normalize=normalize, sense_id=sense_id, word=word, skip_ambigous=skip_ambigous)
                    print("%s\t%s\t%s\t%s" % (word, sense_id,
                        self._format_cluster(ddt_cluster, output_sim), self._format_cluster(ddt_isas, output_sim)), file=output_file)

            print(i+1, "sense clusters processed")
            print(err_num, "senses skipped")
            print("Output file:", output_fpath)

    def _format_cluster(self, ddt_cluster, output_sim):
        if output_sim:
            return ','.join(["%s%s%.3f" % (x[0], SEP3, float(x[1])) for x in sorted(list(ddt_cluster.items()), key=itemgetter(1), reverse=True)])
        else:
            return ','.join([x[0] for x in sorted(list(ddt_cluster.items()), key=itemgetter(1), reverse=True)])

    def _disambiguate(self, target_field, context_fields, sense_fields, sense_id, word, skip_ambigous=False, normalize=True):
        """ disambiguate each cluster/isa word of the input sense"""

        disambiguated_field = {}
        for cword in self._sc[word][sense_id][target_field]:
            try:
                # build context that represents target word
                context = Counter()
                for f in context_fields: context.update(self._sc[word][sense_id][f + "_norm"])

                cword_nopos = cword.split(SEP4)[0]
                cword_lemma = lemmatize_word(cword_nopos)
                context.pop(cword, None)
                context.pop(cword_nopos, None)
                context.pop(cword_lemma, None)
                word_lemma = lemmatize_word(word.split(SEP4)[0])
                context[word_lemma] = 1.0
                context = {w: context[w] for w in context if lemmatize_word(w.split(SEP4)[0]) != cword_lemma}

                # if context word not in the dictionary add ? sense
                self._sense_clusters.find_word(cword_nopos)
                if cword in self._sc:
                    cword_invoc = [cword]
                else:
                    cword_invoc = []
                    cword_invoc += [w for w in self._sense_clusters.find_word(cword_nopos) if w in self._sc]
                    cword_invoc += [w for w in self._sense_clusters.find_word(cword_lemma) if w in self._sc]
                    if len(cword_invoc) == 0 and not skip_ambigous:
                        disambiguated_field[cword + SEP4 + "?"] = self._sc[word][sense_id][target_field][cword]
                        continue

                # if present then select the best sense
                senses = [("?", context, cword)]
                for cwi in cword_invoc:
                    for csense_id in self._sc[cwi]:
                        sense = Counter()
                        for f in sense_fields: sense.update(self._sc[cwi][csense_id][f + "_norm"])
                        sense.pop(cword, None)
                        sense.pop(cword_nopos, None)
                        sense.pop(cword_lemma, None)
                        senses.append((csense_id, sense, cwi))

                dv = DictVectorizer(separator='=', sparse=True)
                X = dv.fit_transform([x[1] for x in senses])
                if normalize:
                    norm = Normalizer(norm=NORM, copy=False)
                    X = norm.transform(X)
                scores = X * X[0, :].T
                scores = scores.toarray()
                scores[0] = 0.0
                best_i = argmax(scores)

                if scores[best_i][0] > MIN_SCORE:
                    disambiguated_field[senses[best_i][2] + SEP4 + str(senses[best_i][0])] = self._sc[word][sense_id][target_field][cword]
                else:
                    if not skip_ambigous: disambiguated_field[cword + SEP4 + "?"] = self._sc[word][sense_id][target_field][cword]

            except KeyboardInterrupt:
                sys.exit()
            except:
                print("Warning:", word, sense_id, cword)
                print(format_exc())
        return disambiguated_field


def run(sense_clusters_fpath, output_fpath, skip_vocabulary_fpath="", skip_ambigous=True):
    s = SenseClusterDisambiguator(sense_clusters_fpath, skip_vocabulary_fpath)
    s.run(output_fpath, skip_ambigous, output_sim=True)


def main():
    parser = argparse.ArgumentParser(description='Make DDT from Sense Clusters.')
    parser.add_argument('sense_clusters', help='Path to a csv file with sense clusters "word<TAB>cid<TAB>cluster<TAB>isas".')
    parser.add_argument('-o', help='Output path. Default -- next to input file.', default="")
    parser.add_argument('-s', help='Skip vocabulary path. Skip these words. Default -- nothing.', default="")
    parser.add_argument('--skip_ambigous', action='store_true', help='Output only disambiguated related words. Default -- True.')
    args = parser.parse_args()

    output_fpath = args.sense_clusters + ".out" if args.o == "" else args.o
    print("Input sense clusters: ", args.sense_clusters)
    print("Output path: ", output_fpath)
    print("Skip vocabulary:", args.s)

    print("Skip ambigous words:", args.skip_ambigous)

    tic = time()
    run(args.sense_clusters, args.s, args.skip_ambigous, output_fpath)
    print(time()-tic, "sec.")


if __name__ == '__main__':
    main()
