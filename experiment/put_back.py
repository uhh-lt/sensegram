""" Evaluate sense vector model on training set"""
# TODO comments

import argparse, codecs
from operator import itemgetter
from gensim.models import word2vec
import wsd, pbar

TEST_WORDS = ['mouse', 'ruby', 'jaguar', 'oracle', 'java']

def window_it(word, file, winsize=10):
    while True:
        line = file.read(10240)
        if not line:
            break
        tokens = line.split()
        for i, token in enumerate(tokens):
            if token == word:
                yield tokens[max(0,i-winsize):i], word , tokens[i+1:i+1+winsize]


def start(sense_path, context_path, text_path, output_path):
    print("Loading models.")
    wsd_model = wsd.WSD(sense_path, context_path)

    with codecs.open(output_path, 'w', encoding='utf-8') as output:
            for test_word in TEST_WORDS:
                senses = wsd.get_senses(wsd_model.vs, test_word)
                contexts = {s: [] for s in senses}

                occur = 0
                # dissambiguate all occurences of test_word
                print("start dissambiguation")
                text = codecs.open(text_path, 'r', encoding='utf-8')
                for l, w, r in window_it(test_word, text):
                    sense, confidence = wsd_model.dis_context(l + r, w)
                    contexts[sense].append((l + r, confidence))
                    if occur%20 == 0:
                        print("Case: %s" % occur) 
                    occur+=1
                text.close()

                # print 100 most confident predictions
                for sense in senses:
                    output.write("Sense: %s. Related words: %s\n" % (sense, wsd_model.vs.most_similar(sense, topn=30)))
                    for context, conf in sorted(contexts[sense], key = itemgetter(1))[:100]:
                        output.write("Confidence: %s. %s\n" % (str(conf), " ".join(context)))

def main():
    parser = argparse.ArgumentParser(description='Evaluate sense vector model on training set')
    parser.add_argument("sense", help="path to a sense vector model")
    parser.add_argument("context", help="path to a context vector model") 
    parser.add_argument("text", help="text on which to evaluate")
    parser.add_argument("output", help="output text file")
    args = parser.parse_args()
    start(args.sense, args.context, args.text, args.output)

if __name__ == '__main__':
    main()