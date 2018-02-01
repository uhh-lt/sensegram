import codecs
from collections import Counter
from operator import itemgetter
import argparse


def run(corpus, output, minsize):
    c = Counter()
    
    with codecs.open(corpus, 'r', encoding='utf-8') as infile:
        for line in infile:
            c.update(line.split())

    with codecs.open(output, 'w', encoding='utf-8') as freqfile:
        for word, freq in sorted(list(c.items()), key=itemgetter(1),  reverse=True):
            if freq >= minsize:
                freqfile.write("%s %i\n" % (word, freq))


def main():
    parser = argparse.ArgumentParser(description='Count term frequencies in tokenized corpus')
    parser.add_argument('corpus', help='A path to a corpus (.txt)')
    parser.add_argument('output', help='A path to the frequencies file')
    parser.add_argument('-minsize', help='Minimum frequency size for terms. Default=20', default=20, type=int)
    args = parser.parse_args()

    run(args.corpus, args.output, args.minsize) 


if __name__ == '__main__':
    main()