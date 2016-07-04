import logging

from numpy import zeros, dtype, float32 as REAL, \
    fromstring
from gensim.models.word2vec import Vocab
from gensim import utils
from gensim.models.word2vec import Word2Vec

logger = logging.getLogger("gensim.models.word2vec")

def load_vectors(fvec, binary):
#    return gs.models.Word2Vec.load_word2vec_format(fvec,binary=True)
    return load_word2vec_format(fvec, binary=binary)


def load_word2vec_format(fname, fvocab=None, binary=False, norm_only=True, encoding='utf8'):
    """
    !!! Code modified from gensim.models.Word2Vec.load_word2vec_format: original version cannot
    load files created by original C word2vec if vocabulary contains words which are not correct
    Unicode byte sequences - this could happen due to corpora encoding issues. !!!

    Load the input-hidden weight matrix from the original C word2vec-tool format.

    Note that the information stored in the file is incomplete (the binary tree is missing),
    so while you can query for word similarity etc., you cannot continue training
    with a model loaded this way.

    `binary` is a boolean indicating whether the data is in binary word2vec format.
    `norm_only` is a boolean indicating whether to only store normalised word2vec vectors in memory.
    Word counts are read from `fvocab` filename, if set (this is the file generated
    by `-save-vocab` flag of the original C tool).

    If you trained the C model using non-utf8 encoding for words, specify that
    encoding in `encoding`.

    """
    counts = None
    if fvocab is not None:
        logger.info("loading word counts from %s" % (fvocab))
        counts = {}
        with utils.smart_open(fvocab) as fin:
            for line in fin:
                word, count = utils.to_unicode(line).strip().split()
                counts[word] = int(count)

    logger.info("loading projection weights from %s" % (fname))
    with utils.smart_open(fname) as fin:
        header = utils.to_unicode(fin.readline(), encoding=encoding)
        vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
        result = Word2Vec(size=vector_size)
        result.syn0 = zeros((vocab_size, vector_size), dtype=REAL)
        if binary:
            binary_len = dtype(REAL).itemsize * vector_size
            for line_no in xrange(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have)

                        word.append(ch)
                try:
                    word = utils.to_unicode(b''.join(word), encoding=encoding)
                except UnicodeDecodeError, e:
                    logger.warning("Couldn't convert whole word to unicode: trying to convert first %d bytes only ..." % e.start)
                    word = utils.to_unicode(b''.join(word[:e.start]), encoding=encoding)
                    logger.warning("... first %d bytes converted to '%s'" % (e.start, word))

                if counts is None:
                    result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
                elif word in counts:
                    result.vocab[word] = Vocab(index=line_no, count=counts[word])
                else:
                    logger.warning("vocabulary file is incomplete")
                    result.vocab[word] = Vocab(index=line_no, count=None)
                result.index2word.append(word)
                result.syn0[line_no] = fromstring(fin.read(binary_len), dtype=REAL)
        else:
            for line_no, line in enumerate(fin):
                parts = utils.to_unicode(line[:-1], encoding=encoding).split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, weights = parts[0], list(map(REAL, parts[1:]))
                if counts is None:
                    result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
                elif word in counts:
                    result.vocab[word] = Vocab(index=line_no, count=counts[word])
                else:
                    logger.warning("vocabulary file is incomplete")
                    result.vocab[word] = Vocab(index=line_no, count=None)
                result.index2word.append(word)
                result.syn0[line_no] = weights
    logger.info("loaded %s matrix from %s" % (result.syn0.shape, fname))
    result.init_sims(norm_only)
    return result
