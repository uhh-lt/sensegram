"""Dependencies required to use of this file can be installed as following:
   pip install gensim clint requests pandas nltk
   python -m nltk.downloader punkt """


import requests
from clint.textui import progress
from os.path import exists
from gensim.models import KeyedVectors
from pandas import read_csv
from nltk.tokenize import word_tokenize
from collections import defaultdict, namedtuple
from operator import itemgetter
from numpy import mean


SenseBase = namedtuple('Sense', 'keyword cluster')

class Sense(SenseBase): # this is needed as list is an unhashable type
    def get_hash(self):
        return hash(self.keyword + "".join(self.cluster))

    def __hash__(self):
        return self.get_hash()

    def __eq__(self, other):
        return self.get_hash() == other.get_hash()


def ensure_word_embeddings(language):
    """ Ensures that the word vectors exist by downloading them if needed. """

    wv_fpath = "cc.{}.300.vec.gz".format(language)
    wv_pkl_fpath = wv_fpath + ".pkl"
    if not exists(wv_fpath):
        wv_uri = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.{}.300.vec.gz".format(language)
        print("Downloading the fasttext model from {}".format(wv_uri))
        r = requests.get(wv_uri, stream=True)
        path = "cc.{}.300.vec.gz".format(language)
        with open(path, "wb") as f:
            total_length = int(r.headers.get("content-length"))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()
    return wv_fpath, wv_pkl_fpath


class WSD(object):
    """ Performs word sense disambiguation based on the induced word senses. """

    def __init__(self, inventory_fpath, language, verbose=False):
        """ :param inventory_fpath path to a CSV file with an induced word sense inventory
            :param language code of the target language of the inventory, e.g. "en", "de" or "fr" """

        _, wv_pkl_fpath = ensure_word_embeddings(language)
        self._wv = KeyedVectors.load(wv_pkl_fpath)
        self._wv.init_sims(replace=True) # normalize the loaded vectors to L2 norm
        self._inventory = self._load_inventory(inventory_fpath)
        self._verbose = verbose
        self._unknown = Sense("UNKNOWN", "")

    def _load_inventory(self, inventory_fpath):
        inventory_df = read_csv(inventory_fpath, sep="\t", encoding="utf-8")

        inventory = defaultdict(lambda: list())
        for i, row in inventory_df.iterrows():
            cluster_words = [cw.strip() for cw in row.cluster.split(",")]
            inventory[row.word].append(Sense(row.keyword, cluster_words))

        return inventory

    def disambiguate(self, target_word, context):
        """ Perform word sense disambiguation: find the correct sense of the target word inside
        the provided context.
        :param target_word an ambigous word that need to be disambiguated
        :param context context of the target_word that allows to disambigute its meaning, represented as a string
        :return sense id of the target_word and a confidence of the prediction """

        try:
            # try to use nltk tokenizer
            tokens = word_tokenize(context)
        except LookupError:
            # do the simple tokenization if not installed
            tokens = context.split(" ")

        return self.disambiguate_tokenized(target_word, tokens)

    def disambiguate_tokenized(self, target_word, tokens, most_significant_num=3):
        """ Perform word sense disambiguation: find the correct sense of the target word inside
        the provided context.
        :param target_word an ambigous word that need to be disambiguated
        :param tokens context of the target_word that allows to disambigute its meaning, represented as a list of tokens
        :param most_significant_num number of the most significant context words which are takein into account from the tokens
        :return sense id of the target_word and a confidence of the prediction """

        # get the inventory
        if target_word not in self._inventory:
            if self._verbose: print("Warning: word '{}' is not in the inventory. ".format(target_word))
            return [(self._unknown, 1.0)]

        senses = self._inventory[target_word]
        if len(senses) == 0:
            if self._verbose: print("Warning: word '{}' has no senses.")
            return [(self._unknown, 1.0)]

        # get vectors of the keywords that represent the senses
        sense_vectors = {}
        for sense in senses:
            if sense.keyword in self._wv.vocab:
                sense_vectors[sense] = self._wv[sense.keyword]
            else:
                print("Warning: keyword '{}' is not in the word embedding model. Skipping the sense.".format(sense.keyword))

        # retrieve vectors of all context words
        context_vectors = {}
        for context_word in tokens:
            is_not_target = not (context_word.lower().startswith(target_word.lower()) and
                                 len(context_word) - len(target_word) <= 1)
            if is_not_target and context_word in self._wv.vocab:
                context_vectors[context_word] = self._wv[context_word]
            else:
                print("Warning: context word '{}' is not in the word embedding model. Skipping the word.".format(context_word))

        # compute distances to all prototypes for each token and pick only those which are discriminative
        context_word_scores = {}
        for context_word in context_vectors:
            scores = []
            for sense in sense_vectors:
                scores.append(context_vectors[context_word].dot(sense_vectors[sense]))
            context_word_scores[context_word] = abs(max(scores) - min(scores))

        best_context_words = sorted(context_word_scores.items(), key=itemgetter(1), reverse=True)[:most_significant_num]

        # average the selected context words
        best_context_vectors = [context_vectors[context_word] for context_word, _ in best_context_words]
        context_vector = mean(best_context_vectors, axis=0)

        # pick the sense which is the most similar to the context vector
        sense_scores = [(sense, context_vector.dot(sense_vectors[sense])) for sense in sense_vectors]
        return sorted(sense_scores, key=itemgetter(1), reverse=True)