from traceback import format_exc
from collections import Counter
from pandas import read_csv
from collections import defaultdict
import pickle as pickle
from utils.common import exists
from utils.common import load_voc
from utils.morph import get_stoplist
from utils.patterns import re_spaced_numbers, re_norm_babel, re_norm_babel_dash, re_whitespaces2
from utils.morph import lemmatize_word


SEP = "\t"
LIST_SEP = ","
SCORE_SEP = ":"
SENSE_SEP = "#"
NORM = "l2"
MIN_SCORE = 0.000001
OUTPUT_UNKNOWN = True
REMOVE_BOW_STOPWORDS = True
LOWERCASE_BOW = True


class SenseClusters(object):
    def __init__(self, sense_clusters_fpath, strip_dst_senses=False, load_sim=True, verbose=False,
                 normalized_bow=False, use_pickle=True, voc_fpath="", voc=[], normalize_sim=False):
        """ Loads and operates sense clusters in the format 'word<TAB>cid<TAB>prob<TAB>cluster<TAB>isas' """

        self._verbose = verbose
        self._normalized_bow = normalized_bow
        self._stoplist = get_stoplist()
        self._normalize_sim = normalize_sim

        if len(voc) > 0:
            self._voc = voc
        elif exists(voc_fpath):
            self._voc = load_voc(voc_fpath)
        else:
            self._voc = {}

        sense_clusters_pkl_fpath = sense_clusters_fpath + ".pkl"
        if use_pickle and exists(sense_clusters_pkl_fpath):
            pkl = pickle.load(open(sense_clusters_pkl_fpath, "rb"))
            if "sense_clusters" in pkl:
                self._sc = pkl["sense_clusters"]
            else:
                print("Error: cannot find sense_clusters in ", sense_clusters_pkl_fpath)
                self._sc = {}

            if "normword2word" in pkl:
                self._normword2word = pkl["normword2word"]
            else:
                print("Error: cannot find normword2word in ", sense_clusters_pkl_fpath)
                self._normword2word = {}
            print("Loaded %d words from: %s" % (len(self._sc), sense_clusters_pkl_fpath))

        else:
            self._sc, self._normword2word = self._load(sense_clusters_fpath, strip_dst_senses, load_sim)
            if use_pickle:
                pkl = {"sense_clusters": self._sc, "normword2word": self._normword2word}
                pickle.dump(pkl, open(sense_clusters_pkl_fpath, "wb"))
                print("Pickled sense clusters:", sense_clusters_pkl_fpath)

    def get_sense_prob(self, word, sense_id):
        """ Returns probability of a word sense. During the first run this
        method can be long as it pre-calculates per word cluster sizes. """

        if word not in self.data or sense_id not in self.data[word]:
            print("Warning: The word sense is absent in the vocabulary of the model. " \
                  "It is not possible to calculate its probability.")
            return 1.0

        recalculation_is_needed = \
            (not hasattr(self, '_cluster_sum')) or \
            (word not in self._cluster_sum)

        if recalculation_is_needed:
            self._calc_cluster_sums()

        cluster_size = float(len(self.data[word][sense_id]["cluster"]))
        if self._cluster_sum[word] > 0:
            sense_prob = cluster_size / self._cluster_sum[word]
        else:
            sense_prob = 1.0

        return sense_prob

    def _calc_cluster_sums(self):
        """ Recalculate cluster sum data structure needed for calculation
         of a priory sense probabilities. """

        self._cluster_sum = defaultdict(int)
        for word in self.data:
            word_sum = 0
            for sense_id in self.data[word]:
                word_sum += len(self.data[word][sense_id]["cluster"])
            self._cluster_sum[word] += word_sum

    def get_num_senses(self, recalculate=False):
        """ Return total number of word senses. """
        if hasattr(self, '_num_senses'):
            if not recalculate:
                # number of senses is known
                pass
            else:
                self._num_senses = self._calc_num_senses()
        else:
            self._num_senses = self._calc_num_senses()

        return self._num_senses

    def _calc_num_senses(self):
        senses_num = 0
        for word in self.data:
            for sense_id in self.data[word]:
                senses_num += 1
        return senses_num

    def _good_token(self, w):
        return (w not in self._stoplist and
                not re_spaced_numbers.match(w))

    @property
    def words(self):
        return list(self._sc.keys())

    @property
    def normwords(self):
        return list(self._normword2word.keys())

    @property
    def data(self):
        return self._sc

    def find_word(self, word):
        return self._normword2word.get(self.norm(word), "")

    def _get_words(self, words_str, strip_dst_senses, load_sim):
        cluster_words = Counter()
        if words_str == "": return cluster_words
        for j, cw in enumerate(words_str.split(LIST_SEP)):
            try:
                fields = cw.split(SCORE_SEP)
                max_sim = 1.0
                if j == 0:
                    max_sim = float(fields[-1]) if len(fields) > 1 else 1.0
                word = fields[0].strip()
                if load_sim:
                    sim = float(fields[-1]) if len(fields) >= 2 else 1.0/(j+1.0)**0.33
                    if self._normalize_sim and sim > 1.0: sim /= max_sim
                else:
                    sim = 1.0

                if strip_dst_senses: word = word.split(SENSE_SEP)[0]

                if not self._normalized_bow or self._good_token(word):
                    cluster_words[word] = float(sim)
            except:
                if self._verbose:
                    print("Warning: bad word '%s'" % cw)
                    print(format_exc())
        return cluster_words

    def _get_normalized_words(self, cluster_words):
        res = {}
        for w in cluster_words:
            token = self.norm(w)
            lemma = lemmatize_word(token)
            if REMOVE_BOW_STOPWORDS and token in self._stoplist or lemma in self._stoplist: continue

            res[lemma] = cluster_words[w]
            res[token] = cluster_words[w]

        return res

    def most_similar(self, word, sense_id=-1, max_number=None, lowercase=False, strip_ids=False):
        if word not in self._sc or len(self._sc[word]) == 0: return []
        senses = self._sc[word]
        res = []
        
        if sense_id == -1:  # return neighbors of all senses
            c = Counter()
            for sid in senses: c.update(senses[sid]["cluster"])
            res = c.most_common(max_number)
        elif sense_id in senses:
            res = senses[sense_id]["cluster"].most_common(max_number)

        if lowercase:
            res = list(set((word.lower(), score) for word, score in res))
       
        if strip_ids:
             res = [(word.split(SENSE_SEP)[0], score) for word, score in res]

        return res

    def norm(self, word):
        if LOWERCASE_BOW: return word.split(SENSE_SEP)[0].lower()
        else: return word.split(SENSE_SEP)[0]

    def _load(self, pcz_fpath, strip_dst_senses, load_sim):
        """ Loads a dict[word][sense] --> {"cluster": Counter(), "cluster_norm": Counter(), "isas": Counter()} """

        senses = defaultdict(dict)
        normword2word = defaultdict(set)
        if not exists(pcz_fpath): return senses, normword2word

        df = read_csv(pcz_fpath, encoding='utf-8', delimiter=SEP, error_bad_lines=True, quotechar='\0')
        df = df.fillna("")
        err_clusters = 0
        num_senses = 0

        # foreach sense cluster
        for i, row in df.iterrows():
            try:
                if i % 25000 == 0: print("%d (%d) senses loaded of %d" % (i, num_senses, len(df)))
                if len(self._voc) > 0 and row.word not in self._voc:
                    continue

                r = {}
                r["prob"] = row.prob if "prob" in row else 1.0
                r["cluster"] = self._get_words(row.cluster, strip_dst_senses, load_sim) if "cluster" in row else Counter()
                r["cluster_norm"] = self._get_normalized_words(r["cluster"]) if self._normalized_bow else r["cluster"]
                r["isas"] = self._get_words(row.isas, strip_dst_senses, load_sim) if "isas" in row else Counter()
                r["isas_norm"] =  self._get_normalized_words(r["isas"]) if self._normalized_bow else r["isas"]
                senses[row.word][row.cid] = r
                normword2word[self.norm(row.word)].add(row.word)
                num_senses += 1
            except:
                print(".", end=' ')
                if self._verbose:
                    print("Warning: bad cluster")
                    print(row)
                    print(format_exc())
                err_clusters += 1

        print(err_clusters, "cluster errors")
        print(num_senses, "senses loaded out of", i + 1)
        print(len(senses), "words loaded")

        return senses, normword2word


    def _normalize(self, word, dash=False):
        word = re_norm_babel_dash.sub(" ", word) if dash else re_norm_babel.sub(" ", word)
        word = re_whitespaces2.sub(" ", word)
        return word.lower().strip()

    def _filter_cluster(self, cluster):
        return Counter({w: cluster[w] for w in cluster if self._good_token(w)})

    def get_senses_full(self, word):
        return self._sc[word] if word in self._sc else []

    def get_senses(self, word, min_prob=0.0):
        """ Returns a list of tuples (sense_id, bow), where bow is a Counter and sense_id is a unicode """

        if word not in self._sc:
            return []
        else:
            field = "cluster_norm" if self._normalized_bow else "cluster"
            return [(str(cid), self._filter_cluster(self._sc[word][cid][field]))
                    for cid in self._sc[word] if self._sc[word][cid]["prob"] > min_prob]

    def get_cluster(self, word, sense_id):
        """ Returns cluster of a given word sense  """

        field = "cluster_norm" if self._normalized_bow else "cluster"

        if word in self._sc and sense_id in self._sc[word]:
            return self._sc[word][sense_id][field]
        elif word in self._sc and str(sense_id) in self._sc[word]:
            return self._sc[word][str(sense_id)][field]
        elif word in self._sc:
            try:
                return self._sc[word][int(sense_id)][field]
            except:
                return Counter()
        else: return Counter()
