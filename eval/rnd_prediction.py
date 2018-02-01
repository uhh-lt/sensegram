import argparse
from pandas import read_csv
from csv import QUOTE_NONE
from random import randint
from sensegram import SenseGram


def run(test_file, vs, output):
    print("Loading test set...")
    reader = read_csv(test_file, encoding="utf-8", delimiter="\t", dtype={'predict_related': object, 'gold_sense_ids':object, 'predict_sense_ids':object})
    rows_count = reader.shape[0]
    print((str(rows_count) + " test instances"))
    
    for i, row in reader.iterrows():
        sense_count = len(vs.get_senses(row.target))
        if sense_count > 0:
            rand_sense = randint(0, sense_count-1)
            reader.set_value(i, 'predict_sense_ids', rand_sense)
        else: 
            print(("0 senses for " + row.target))
    
    reader.to_csv(sep='\t', path_or_buf=output, encoding="utf-8", index=False, quoting=QUOTE_NONE)
    print(("Saved predictions to " + output))
    

def main():
    parser = argparse.ArgumentParser(description='Fill in a test dataset using Random Sense method.')
    parser.add_argument('test_file', help='A path to a test dataset. Format: "context_id<TAB>target<TAB>target_pos<TAB>target_position<TAB>gold_sense_ids<TAB>predict_sense_ids<TAB>golden_related<TAB>predict_related<TAB>context')
    parser.add_argument("senses", help="A path to sense vectors")
    parser.add_argument("output", help="An output path to the filled dataset. Same format as test_file")
    
    args = parser.parse_args()
    
    print("Loading sense model...")
    vs = SenseGram.load_word2vec_format(args.senses, binary=False)
    run(args.test_file, vs, args.output)


if __name__ == '__main__':
    main()
