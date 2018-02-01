import argparse
from pandas import read_csv
from csv import QUOTE_NONE
from sensegram import SenseGram
from wsd import WSD
from gensim.models import word2vec
from utils import pbar


NEIGHBORS_NUM = 50


def run(test_file, sense, context, output, wsd_method="sim", filter_ctx=2, lowercase=False, ignore_case=False):
    print("Loading models...")
    vs = SenseGram.load_word2vec_format(sense, binary=False)
    vc = word2vec.Word2Vec.load_word2vec_format(context, binary=False)
    wsd_model = WSD(vs, vc, method=wsd_method, filter_ctx=filter_ctx, ignore_case=ignore_case)

    print("Loading test set...")
    reader = read_csv(test_file, encoding="utf-8", delimiter="\t", dtype={'predict_related': object, 'gold_sense_ids':object, 'predict_sense_ids':object})
    rows_count = reader.shape[0]
    print((str(rows_count) + " test instances"))
    pb = pbar.Pbar(rows_count, 100)
    

    uncovered_words = [] # target words for which sense model has zero senses

    print(("Start prediction over " + test_file))
    pb.start()
    for i, row in reader.iterrows():
        # Form of prediction: (sense, sense_scores)
        ctx = row.context.lower() if lowercase else row.context
        start, end = [int(x) for x in row.target_position.split(',')]
        
        prediction = wsd_model.dis_text(ctx, row.target, start, end)
        if prediction:
            sense, sense_scores = prediction
            reader.set_value(i, 'predict_sense_ids', sense.split("#")[1])
                #neighbours = wsd_model.vs.most_similar(sense, topn=n_neighbours)
                #neighbours = ["%s:%.3f" % (n.split("#")[0], float(sim)) for n, sim in neighbours]
                #reader.set_value(i, 'predict_related', ",".join(neighbours))
        else:
            uncovered_words.append(row.target)
            continue
            
        pb.update(i)
    pb.finish()
    
    reader.to_csv(sep='\t', path_or_buf=output, encoding="utf-8", index=False, quoting=QUOTE_NONE)
    print(("Saved predictions to " + output))


def main():
    parser = argparse.ArgumentParser(description='Fill in a test dataset for word sense disambiguation.')
    parser.add_argument('test_file', help='A path to a test dataset. Format: "context_id<TAB>target<TAB>target_pos<TAB>target_position<TAB>gold_sense_ids<TAB>predict_sense_ids<TAB>golden_related<TAB>predict_related<TAB>context')
    parser.add_argument("sense", help="A path to a sense vector model")
    parser.add_argument("context", help="A path to a context vector model")
    parser.add_argument("output", help="An output path to the filled dataset. Same format as test_file")
    parser.add_argument("-wsd_method", help="WSD method 'prob' or 'sim'. Default='sim'", default="sim")
    parser.add_argument("-filter_ctx", help="Number of context words for WSD (-1 for no filtering). Default is 2.", default=2, type=int)
    parser.add_argument("-lowercase_context", help="Lowercase all words in context (necessary if context vector model only has lowercased words). Default False", action="store_true")
    parser.add_argument("-ignore_case", help="Ignore case of a target word (consider upper- and lower-cased senses). Default False", action="store_true")
    args = parser.parse_args()

    run(args.test_file, args.sense, args.context, args.output, args.wsd_method, args.filter_ctx, args.lowercase_context, args.ignore_case) 


if __name__ == '__main__':
    main()
