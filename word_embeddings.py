import gzip
import codecs
from gensim.utils import tokenize
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from time import time
from os import listdir
from os.path import exists, isdir, join
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
from collections import defaultdict


class GzippedCorpusStreamer(object):
    def __init__(self, corpus_fpath):
        self._corpus_fpath = corpus_fpath
        
    def __iter__(self):
        if isdir(self._corpus_fpath):
            for fname in listdir(self._corpus_fpath):
                corpus_fpath = join(self._corpus_fpath, fname)
                print("Reading from file:", corpus_fpath)
                yield from self._read_file(corpus_fpath)
        else:
            print("Reading from file:", self._corpus_fpath)
            yield from self._read_file(self._corpus_fpath)

    def _read_file(self, corpus_fpath):
        if corpus_fpath.endswith(".txt.gz"):
            corpus = gzip.open(corpus_fpath, "r", "utf-8")
        else:
            corpus = codecs.open(corpus_fpath, "r", "utf-8")
        for line in corpus:
            yield list(tokenize(line,
                                lowercase=False,
                                deacc=False,
                                encoding='utf8',
                                errors='strict',
                                to_lower=False,
                                lower=False))


class PhraseDetector(object):
    def __init__(self, vocabulary_fpath, do_restore_bigrams=True):
        self._restore_bigrams = do_restore_bigrams
        self._phrases = self._load_vocabulary(vocabulary_fpath)
        self._ngram_max = self._get_ngram_max(self._phrases)
        self._stats = defaultdict(int)
        for p in self._phrases:
            self._stats[p] = 0

    def print_stats(self):
        for phrase in self._stats:
            print("phrase:\t{}\t{}".format(phrase, self._stats[phrase]))

    def _load_vocabulary(self, vocabulary_fpath):
        voc = set()
        with codecs.open(vocabulary_fpath, "r", "utf-8") as voc_file:
            for line in voc_file:
                word = line.strip()
                voc.add(word)
                voc.add(word.capitalize())
                word = word.replace(" ", "_")
                voc.add(word)
                voc.add(word.capitalize())

        return voc

    def _get_ngram_max(self, phrases):
        max_len = 0

        for p in phrases:
            p_len = len(p.split("_"))
            if max_len < p_len: max_len = p_len

        return max_len

    def _split_tokens(self, tokens):
        splitted_tokens = []

        for t in tokens:
            if "_" in t:
                splitted = t.split("_")
                for st in splitted:
                    splitted_tokens.append(st)
            else:
                splitted_tokens.append(t)

        return splitted_tokens

    def _add_dict_phrases(self, tokens):
        splitted_tokens = self._split_tokens(tokens) if self._restore_bigrams else tokens

        phrase_tokens = []
        skip_tokens = 0

        for i in range(len(splitted_tokens)):
            if skip_tokens > 0:
                skip_tokens -= 1
                continue

            for ngram_size in range(self._ngram_max, 2 - 1, -1):
                phrase_candidate = "_".join(splitted_tokens[i:i + ngram_size])
                if phrase_candidate in self._phrases:
                    phrase_tokens.append(phrase_candidate)
                    self._stats[phrase_candidate] += 1
                    skip_tokens = ngram_size - 1
                    print("+++", phrase_candidate)
                    break

            if skip_tokens == 0:
                phrase_tokens.append(splitted_tokens[i])

        return phrase_tokens

    def _get_bigrams(self, tokens):
        bigrams = set()

        for t in tokens:
            if "_" in t: bigrams.add(t)

        return bigrams

    def _restore_bigrams(self, tokens_with_phrases, tokens_with_bigrams):
        bigrams = self._get_bigrams(tokens_with_bigrams)

        tokens_with_phrases_and_bigrams = []
        skip = False
        for i in range(len(tokens_with_phrases)):
            if skip:
                skip = False
                continue

            bigram_candidate_space = " ".join(tokens_with_phrases[i:i + 2])
            bigram_candidate_under = "_".join(tokens_with_phrases[i:i + 2])

            if "_" not in bigram_candidate_space and bigram_candidate_under in bigrams:
                tokens_with_phrases_and_bigrams.append(bigram_candidate_under)
                skip = True
            else:
                tokens_with_phrases_and_bigrams.append(tokens_with_phrases[i])

        return tokens_with_phrases_and_bigrams

    def add_phrases(self, tokens):
        """ Add multiword phrases to the input sequence of tokens. """

        tokens_with_phrases = self._add_dict_phrases(tokens)

        if self._restore_bigrams:
           return self._restore_bigrams(tokens_with_phrases, tokens)
        else:
            return tokens_with_phrases


def detect_phrases(corpus_fpath, phrases_fpath, batch_size=500000):
    """ Gets a text corpus as input, detect phrases and saves an updated corpus to filesystem.
    The path to the resulting corpus is returned from this function. """

    output_fpath = corpus_fpath + ".phrases.txt.gz"
    sentences = GzippedCorpusStreamer(corpus_fpath)
    pd = PhraseDetector(phrases_fpath, do_restore_bigrams=False)
    pool = Pool(processes=cpu_count())

    s_batch = []
    with gzip.open(output_fpath, "wt", encoding="utf-8") as out:
        for s in tqdm(sentences):

            s_batch.append(s)
            if len(s_batch) == batch_size:
                for s in pool.map(pd.add_phrases, s_batch):
                    out.write("{}\n".format(" ".join(s)))

                s_batch = []

        for s in pool.map(pd.add_phrases, s_batch): out.write("{}\n".format(" ".join(s)))


    pd.print_stats()

    return output_fpath


def learn_word_embeddings(corpus_fpath, vectors_fpath, cbow, window, iter_num, size, threads,
                          min_count, detect_bigrams=True, phrases_fpath=""):

    tic = time()

    if exists(phrases_fpath):
        tic = time()
        print("Finding phrases from the input dictionary:", phrases_fpath)
        corpus_fpath = detect_phrases(corpus_fpath, phrases_fpath, batch_size=500000)
        print("Time, sec.: {}".format(time() - tic))

    sentences = GzippedCorpusStreamer(corpus_fpath)
    
    if detect_bigrams:
        print("Extracting bigrams from the corpus:", corpus_fpath)

        bigram_transformer = Phrases(sentences, min_count=min_count)
        bigrams = Phraser(bigram_transformer)
        sentences = list(bigrams[sentences])
        print("Time, sec.:", time() - tic)

    print("Training word vectors:", corpus_fpath)
    model = Word2Vec(sentences,
                     min_count=min_count,
                     size=size,
                     window=window, 
                     max_vocab_size=None,
                     workers=threads,
                     sg=(1 if cbow == 0 else 0),
                     iter=iter_num)
    model.wv.save_word2vec_format(vectors_fpath, binary=False)
    print("Vectors:", vectors_fpath)
    print("Time, sec.:", time()-tic) 

