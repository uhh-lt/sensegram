#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" """

import argparse, codecs
from pandas import read_csv
from wsd import WSD
import pbar

debug=False
n_neighbours = 50

def use_prediction(pr, entropy_thr, diff_thr):
    """ Decide what to do with prediction depending on confidence thresholds
    """
    sense, distrib, entr, diff, ctx_len = pr
    if len(distrib) == 1:
        return True
        # think how to handle a situation, if there is only 1 sense to choose.
        # Entropy and Diff are not appropriate.
        # Entropy of [.] of always 0 (very confident)
        # Diff of [.] is always 0 (very unconfident)
    if not(entropy_thr or diff_thr):
        return True 
    if entropy_thr:
        return True if entr <= float(entropy_thr) else False
    if diff_thr:
        return True if diff >= float(diff_thr) else False 

def run(test_file, sense, context, output, entropy_thr=None, diff_thr=None, lowercase=False):
    print("Loading models...")
    wsd_model = WSD(sense, context)

    print("Loading test set...")
    reader = read_csv(test_file, encoding="utf-8", delimiter="\t", dtype={'predict_related': object, 'gold_sense_ids':object, 'predict_sense_ids':object})
    rows_count = reader.shape[0]
    print(unicode(rows_count) + " test instances")
    pb = pbar.Pbar(rows_count, 100)
    

    uncovered_words = [] # target words for which sense model has zero senses
    entropies = [] # all observed entropy metric values, with corresponding number of senses, and words in context
    diffs = [] # all observed difference metric values, with corresponding number of senses, words in context and a probability of the chosen sense

    print("Start prediction over " + test_file)
    pb.start()
    for i, row in reader.iterrows():
        # Form of prediction: (sense, distrib, e_conf, diff_conf, ctx_len)
        ctx = row.context.lower() if lowercase else row.context
        prediction = wsd_model.dis_text(ctx, row.target_position, row.target)
        if prediction:
            sense, distrib, entr, diff, ctx_len = prediction
            entropies.append((entr, len(distrib), ctx_len))
            diffs.append((diff, len(distrib), ctx_len, max(distrib)))
            
            if use_prediction(prediction, entropy_thr, diff_thr):
                reader.set_value(i, 'predict_sense_ids', sense.split("#")[1])
                neighbours = wsd_model.vs.most_similar(sense, topn=n_neighbours)
                neighbours = ["%s:%.3f" % (n.split("#")[0], float(sim)) for n, sim in neighbours]
                reader.set_value(i, 'predict_related', ",".join(neighbours))
        else:
            uncovered_words.append(row.target)
            continue
        pb.update(i)
    pb.finish()

    reader.to_csv(sep='\t', path_or_buf=output, encoding="utf-8", index=False)
    print("Saved predictions to " + output)

    if debug:
        with codecs.open(output + ".stat", 'w', encoding="utf-8") as conf:
            conf.write(u"Entropy\tn_senses\tcontext_len\tDifference\tn_senses\tcontext_len\tprob\n")
            # low entropy -> high confidence
            entropies = [("%s\t%s\t%s" % row) for row in sorted(entropies)]
            # high diff -> high confidence
            diffs = [("%s\t%s\t%s\t%s" % row) for row in sorted(diffs, reverse=True)]
            for entr, diff in zip(entropies, diffs):
                conf.write(entr + u"\t" + diff + u"\n")
            conf.write("\nUncovered target words: \n")
            conf.write("\n".join(uncovered_words))
        print("Saved statistics to " + output + ".stat")


            # File has two different tables: entropies and difference.
            # Tables are independent! They show distribution of these two metrics over instances in test file
            # Rows in tables are sorted by decreasing confidence of prediction.
            # They do not follow the order of instances in test file.
            # For each row: n_senses - how many possible senses were considered,
            # context_len - how many words was there in context
            # separately, probability of the chosen sense is printed, bound to Difference table
            # TODO output unconfident predictions



def main():
    parser = argparse.ArgumentParser(description='Fill in a test dataset for word sense disambiguation.')
    parser.add_argument('test_file', help='A path to a test dataset. Format: "context_id<TAB>target<TAB>target_pos<TAB>target_position<TAB>gold_sense_ids<TAB>predict_sense_ids<TAB>golden_related<TAB>predict_related<TAB>context')
    parser.add_argument("sense", help="A path to a sense vector model")
    parser.add_argument("context", help="A path to a context vector model")
    parser.add_argument("output", help="An output path to the filled dataset. Same format as test_file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-entropy_thr", help="A threshold for an entropy metric (lower value -> higher confidence). Default=None", default=None)
    group.add_argument("-diff_thr", help="A threshold for a difference metric (lower value -> higher confidence). Default=None", default=None)
    parser.add_argument("-lowercase", help="Lowercase all words in context (necessary if context vector model only has lowercased words). Default False", action="store_true")
    args = parser.parse_args()

    run(args.test_file, args.sense, args.context, args.output, args.entropy_thr, args.diff_thr, args.lowercase) 
    
if __name__ == '__main__':
    main()
