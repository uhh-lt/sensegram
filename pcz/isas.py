from pandas import read_csv
import pickle as pickle
from utils.common import exists, preprocess_pandas_csv
from utils.common import try_remove
from traceback import format_exc
from collections import defaultdict
import operator
from utils.morph import lemmatize


MAX_ISAS = 1000


class ISAs(object):
    def __init__(self, isas_fpath, min_freq=0.0, preprocess=True, sep='\t', strip_pos=True, use_pickle=True, lowercase=True):
        """ Provides access to a ISAs relations from a CSV file "hyponym<TAB>hypernym<TAB>freq" """

        if not exists(isas_fpath):
            self._hypo2hyper = {}
            self._hyper2hypo = {}
            return

        isas_pkl_fpath = isas_fpath + ".pkl"
        if use_pickle and exists(isas_pkl_fpath):
            pkl = pickle.load(open(isas_pkl_fpath, "rb"))
            if "hypo2hyper" in pkl:
                hypo2hyper = pkl["hypo2hyper"]
            else:
                print(("Error: cannot find hypo2hyper in ", isas_pkl_fpath))
                hypo2hyper = {}

            if "hyper2hypo" in pkl:
                hyper2hypo = pkl["hyper2hypo"]
            else:
                print(("Error: cannot find hyper2hypo in ", isas_pkl_fpath))
                hyper2hypo = {}

        else:
            if preprocess:
                isas_cln_fpath = isas_fpath + ".cleaned"
                preprocess_pandas_csv(isas_fpath, isas_cln_fpath)
                isas_df = read_csv(isas_cln_fpath, sep, encoding='utf8', error_bad_lines=False)
                try_remove(isas_cln_fpath)
            else:
                isas_df = read_csv(isas_fpath, sep, encoding='utf8', error_bad_lines=False)

            isas_df = isas_df.drop(isas_df[isas_df["freq"] < min_freq].index)
            hypo2hyper = defaultdict(dict)
            hyper2hypo = defaultdict(dict)
            for i, row in isas_df.iterrows():
                try:
                    hypo = str(row["hyponym"]).split("#")[0].lower() if lowercase else str(row["hyponym"]).split("#")[0]
                    hyper = str(row["hypernym"]).split("#")[0].lower() if lowercase else str(row["hypernym"]).split("#")[0]
                    freq = float(row["freq"])
                    hypo_lemma = lemmatize(hypo).lower()
                    hyper_lemma = lemmatize(hyper).lower()

                    # hypo2hyper
                    if hypo not in hypo2hyper or hyper not in hypo2hyper[hypo]: hypo2hyper[hypo][hyper] = freq
                    else: hypo2hyper[hypo][hyper] += freq
                    if (hypo_lemma, hyper_lemma) != (hypo, hyper): 
                        if hypo_lemma not in hypo2hyper or hyper_lemma not in hypo2hyper[hypo_lemma]: hypo2hyper[hypo_lemma][hyper_lemma] = freq
                        else: hypo2hyper[hypo_lemma][hyper_lemma] += freq
                    
                    # hyper2hypo
                    if hyper not in hyper2hypo or hypo not in hyper2hypo[hyper]: hyper2hypo[hyper][hypo] = freq
                    else: hyper2hypo[hyper][hypo] += freq
                    if (hypo_lemma, hyper_lemma) != (hypo, hyper):
                        if hyper_lemma not in hyper2hypo or hypo_lemma not in hyper2hypo[hyper_lemma]: hyper2hypo[hyper_lemma][hypo_lemma] = freq
                        else: hyper2hypo[hyper_lemma][hypo_lemma] += freq

                except:
                    print(("Bad row:", row))
                    print((format_exc()))

            print(("dictionary is loaded:", len(hypo2hyper)))

            if use_pickle:
                pkl = {"hypo2hyper": hypo2hyper, "hyper2hypo": hyper2hypo}
                pickle.dump(pkl, open(isas_pkl_fpath, "wb"))
                print(("Pickled voc:", isas_pkl_fpath))

        print(("Loaded %d words from: %s" % (len(hypo2hyper), isas_pkl_fpath if isas_pkl_fpath else isas_fpath)))

        self._hypo2hyper = hypo2hyper
        self._hyper2hypo = hyper2hypo

    @property
    def data(self):
        return self._hypo2hyper

    @property
    def hypo2hyper(self):
        return self._hypo2hyper

    @property
    def hyper2hypo(self):
        return self._hyper2hypo

    def has_isa(self, hypo, hyper):
        return self.has_relation(hypo, hyper)

    def has_relation(self, hypo, hyper):
        hypo = str(hypo)
        hyper = str(hyper)

        hypo_variants = set([hypo, hypo.lower(), lemmatize(hypo), lemmatize(hypo).lower()])
        hyper_variants = set([hyper, hyper.lower(), lemmatize(hyper), lemmatize(hyper).lower()])
        freqs = [0]

        for w in hypo_variants:
            for iw in hyper_variants:
                if w in self._hypo2hyper and iw in self._hypo2hyper[w]: freqs.append(self._hypo2hyper[w][iw])

        return max(freqs)

    def all_isas(self, word):
        return self.all_hyper(word)

    def all_hypo(self, hyper, max_output=MAX_ISAS):
        """ Returns all hypo relations of a hyper """

        hyper = str(hyper)

        if hyper in self._hyper2hypo:
            res = self._hyper2hypo[hyper]
        elif hyper.lower() in self._hyper2hypo:
            res = self._hyper2hypo[hyper.lower()]
        else:
            res = {}
        
        res.pop(hyper, None)
        res_sort = sorted(list(res.items()), key=operator.itemgetter(1), reverse=True)
        return res_sort[:min(len(res_sort),max_output)]

    def all_hyper(self, hypo, max_output=MAX_ISAS):
        """ Returns all hyper relations of a hypo """
        
        hypo = str(hypo)

        if hypo in self._hypo2hyper:
            res = self._hypo2hyper[hypo]
        elif hypo.lower() in self._hypo2hyper:
            res = self._hypo2hyper[hypo.lower()]
        else:
            res = {}

        res.pop(hypo, None)
        res_sort = sorted(list(res.items()), key=operator.itemgetter(1), reverse=True)
        return res_sort[:min(len(res_sort),max_output)]

