"""dependencies required to use of this file:
pip install gensim clint requests pandas """

import requests
from clint.textui import progress
from os.path import exists
from gensim.models import KeyedVectors
from pandas import read_csv


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
        wv = KeyedVectors.load(wv_pkl_fpath)
        inventory_df = read_csv(inventory_fpath, sep="\t", encoding="utf-8")
        # convert inventory into somehting more digestable e.g. dictionary word -> sense information


    def disambiguate(self, target_word, context, context_is_tokenized=False):
        """ wsd """

        if context_is_tokenized:
            tokens = context
        else:
            tokens = context.split(" ") # or something more involved ... if the language is European


