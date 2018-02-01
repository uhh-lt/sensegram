import gzip, codecs, argparse
from nltk.tokenize import word_tokenize


def run(corpus, output):
    if corpus.endswith(".gz"):  
        with gzip.open(corpus) as fp, codecs.open(output, mode='w', encoding='utf-8') as out:
            for byteline in fp: # bytestring now has the uncompressed bytes of foo.gz
                line = byteline.decode('utf-8')
                out.write(" ".join(word_tokenize(line)) + '\n')
    else:
        with codecs.open(corpus, mode='r', encoding='utf-8') as fp, codecs.open(output, mode='w', encoding='utf-8') as out:
            for line in fp: # bytestring now has the uncompressed bytes of foo.gz
                out.write(" ".join(word_tokenize(line)) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Tokenize corpus for word2vec (Treebank tokenizer)')
    parser.add_argument('corpus', help='A path to a corpus (.gz or .txt)')
    parser.add_argument('output', help='A path to the the processed corpus')
    args = parser.parse_args()

    run(args.corpus, args.output) 


if __name__ == '__main__':
    main()