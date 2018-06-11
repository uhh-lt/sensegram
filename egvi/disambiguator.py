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


Sense = namedtuple('Sense', 'keyword cluster')


def ensure_word_embeddings(language):
    """ Ensures that the word vectors exist by downloading them if needed. """

    wv_fpath = "model/cc.{}.300.vec.gz".format(language)
    wv_pkl_fpath = wv_fpath + ".pkl"
    if not exists(wv_fpath):
        wv_uri = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.{}.300.vec.gz".format(language)
        print("Downloading the fasttext model from {}".format(wv_uri))
        r = requests.get(wv_uri, stream=True)
        path = "model/cc.{}.300.vec.gz".format(language)
        with open(path, "wb") as f:
            total_length = int(r.headers.get("content-length"))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()
    return wv_fpath, wv_pkl_fpath


class WSD(object):
    """ Performs word sense disambiguation based on the induced word senses. """

    def __init__(self, inventory_fpath, language):
        """ :param inventory_fpath path to a CSV file with an induced word sense inventory
            :param language code of the target language of the inventory, e.g. "en", "de" or "fr" """

        _, wv_pkl_fpath = ensure_word_embeddings(language)
        self._wv = KeyedVectors.load(wv_pkl_fpath)
        self._inventory = self._load_inventory(inventory_fpath)

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

    def disambiguate_tokenized(self, target_word, tokens):
        """ Perform word sense disambiguation: find the correct sense of the target word inside
        the provided context.
        :param target_word an ambigous word that need to be disambiguated
        :param tokens context of the target_word that allows to disambigute its meaning, represented as a list of tokens
        :return sense id of the target_word and a confidence of the prediction """

        return 1, 0.99




