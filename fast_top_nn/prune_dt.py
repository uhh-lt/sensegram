import argparse, codecs
from spacy.en import English
from collections import defaultdict
from pandas import read_csv


STOPWORDS = "context-eval/data/stopwords.csv"

def load_stoplist(fpath):
    word_df = read_csv(fpath, sep="\t", quotechar="\0",doublequote=False,  encoding='utf8', error_bad_lines=False)
    voc = set(row["word"] for i, row in word_df.iterrows())
    print(("loaded %d stopwords: %s" % (len(voc), fpath)))
    return voc

def load_freq(freq_file):
    print("Loading frequencies")
    d = defaultdict(int)
    with codecs.open(freq_file, 'r', encoding='utf-8') as f:
        for line in f:
            (key, val) = line.split()
            d[key] = int(val)
    return d

def lemmatize(text):
    tokens = _spacy(text, tag=True, parse=False, entity=True)
    text_lemmatized = " ".join(t.lemma_ for t in tokens)
    return text_lemmatized

def run(infile, freq_file, outfile, n, stop):
    fr = load_freq(freq_file)
    if stop:
        print("Loading stop words and spicy model")
        _stopwords = load_stoplist(STOPWORDS)
        global _spacy
        _spacy = English()
        
    print("Iterating through input file")
    with codecs.open(infile, "r", encoding="utf-8") as inDT, codecs.open(outfile, "w", encoding="utf-8") as output:
        for line in inDT:
            #try:
            f = True
            (word1, word2, sim) = line.split('\t')
            if fr[word1] < n or fr[word2] < n:
                f = False
            if stop and word2 in _stopwords and word1 not in _stopwords:
                f = False
            if stop and lemmatize(word1) == lemmatize(word2):
                f = False
            if f:
                output.write("%s\t%s\t%s" % (word1, word2, sim))
            #except: 
                #print "Parse problem in line: ", line

def main():
    parser = argparse.ArgumentParser(description='Prune DT file (word<Tab>word<Tab>similarity) to decrease size. Delete pairs with words whose frequency is lower than n.')
    parser.add_argument('infile', help='Path to file to prune')
    parser.add_argument('freqs', help="Path to file with word frequencies. Format word<whitespace>count")
    parser.add_argument('output', help='Path to output file (same format)')
    parser.add_argument('n', help="Frequency threshold for words", type=int)
    parser.add_argument('-stop', help="Delete stop words and flexed forms from cluster", action='store_true')
    args = parser.parse_args()
    
    print(("Input DT:", args.infile))
    print(("Frequencies file:", args.freqs))
    print(("Pruned DT:", args.output))
    print(("Frequency threshold:", args.n))
    
    run(args.infile, args.freqs, args.output, args.n, args.stop) 
    
if __name__ == '__main__':
    main()