import argparse, codecs
from collections import defaultdict

def load_freq(freq_file):
    print "Loading frequencies"
    d = defaultdict(int)
    with codecs.open(freq_file, 'r', encoding='utf-8') as f:
        for line in f:
            (key, val) = line.split()
            d[key] = int(val)
    return d

def run(infile, freq_file, outfile, n):
    fr = load_freq(freq_file)
    
    print "Iterating through input file"
    with codecs.open(infile, "r", encoding="utf-8") as inDT, codecs.open(outfile, "w", encoding="utf-8") as output:
        for line in inDT:
            try:
                (word1, word2, sim) = line.split('\t')
                if fr[word1] >= n and fr[word2] >= n:
                    output.write("%s\t%s\t%s" % (word1, word2, sim))
            except: 
                print "Parse problem in line: ", line
    

def main():
    parser = argparse.ArgumentParser(description='Prune DT file (word<Tab>word<Tab>similarity) to decrease size. Delete pairs with words whose frequency is lower than n.')
    parser.add_argument('infile', help='Path to file to prune')
    parser.add_argument('freqs', help="Path to file with word frequencies. Format word<whitespace>count")
    parser.add_argument('output', help='Path to output file (same format)')
    parser.add_argument('n', help="Frequency threshold for words", type=int)
    args = parser.parse_args()
    
    print "Input DT:", args.infile
    print "Frequencies file:", args.freqs
    print "Pruned DT:", args.output
    print "Frequency threshold:", args.n
    
    run(args.infile, args.freqs, args.output, args.n) 
    
if __name__ == '__main__':
    main()