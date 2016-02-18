#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Evaluate sense vector model on training set"""
# TODO comments

import argparse, codecs
from operator import itemgetter
from gensim.models import word2vec
import wsd, pbar

TEST_WORDS = ['mouse', 'ruby', 'jaguar', 'oracle', 'java']
OUTPUT_HEADER = "word\tsense_id\trelated_terms\tdistr\te_conf\tdiff_conf\tctx_len\tcontext\n"
N_REL_TERMS = 20

def window_it(word, file, lowercase, winsize=10):
    while True:
        line = file.read(10240)
        if not line:
            break
        if lowercase:   
            tokens = line.lower().split()
        else:
            tokens = line.split()
        for i, token in enumerate(tokens):
            if token == word:
                yield tokens[max(0,i-winsize):i], word , tokens[i+1:i+1+winsize]


def run(sense_path, context_path, text_path, output_path, lowercase=False,
        test_words=TEST_WORDS):
    print("Loading models.")
    wsd_model = wsd.WSD(sense_path, context_path)

    with codecs.open(output_path, 'w', encoding='utf-8') as output:
            output.write(OUTPUT_HEADER)
            for test_word in test_words:
                senses = wsd.get_senses(wsd_model.vs, test_word)
                if len(senses) > 0:
                    rel_terms = {s: wsd_model.vs.most_similar(positive=[s],topn=N_REL_TERMS) for s in senses}
                    contexts = {s: [] for s in senses}

                    occur = 0
                    # disambiguate all occurences of test_word
                    print("Start disambiguation for " + test_word)
                    text = codecs.open(text_path, 'r', encoding='utf-8')
                    for l, w, r in window_it(test_word, text, lowercase):
                        sense, distrib, e_conf, diff_conf, ctx_len = wsd_model.dis_context(l + r, w)
                        contexts[sense].append((distrib, e_conf, diff_conf, ctx_len, " ".join(l) + " " + w + " " + " ".join(r)))
                        if occur%20 == 0:
                            print("Case: %s" % occur) 
                        occur+=1
                    text.close()

                    # print predictions
                    for sense in senses:
                        word, sense_id = sense.split("#")
                        terms = ",".join([term + (u":%.3f" % sim) for term, sim in rel_terms[sense]])
                        # sort instances by entropy confidence. Lower value = better confidence. Best are at the beginning. 
                        for distrib, e_conf, diff_conf, ctx_len, context in sorted(contexts[sense], key = itemgetter(1)):
                            output.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (word, sense_id, terms, distrib, e_conf, diff_conf, ctx_len, context))

def main():
    parser = argparse.ArgumentParser(description='Evaluate sense vector model on training set')
    parser.add_argument("sense", help="path to a sense vector model")
    parser.add_argument("context", help="path to a context vector model") 
    parser.add_argument("text", help="text on which to evaluate")
    parser.add_argument("output", help="output text file")
    parser.add_argument("-lowercase", help="Lowercase all words in context (necessary if sense vector model only has lowercased words). Default False", action="store_true")
    parser.add_argument("-words", help="Words to be disambiguated. Use ',' as separator. Default = 'mouse,ruby,jaguar,oracle,java'", default=TEST_WORDS)

    args = parser.parse_args()
    run(args.sense, args.context, args.text, args.output, args.lowercase, args.words.split(','))

if __name__ == '__main__':
    main()