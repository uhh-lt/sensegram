import argparse
from pandas import read_csv
from csv import QUOTE_NONE
import numpy as np


def mfs_mapping(inventory):
    mapping = {}
    print("Loading provided inventory " + inventory)

    inv = read_csv(inventory, sep="\t", encoding='utf8', header=None,
            names=["word","sense_id","cluster"], dtype={'sense_id':np.str, 'cluster':np.str}, 
            doublequote=False, quotechar="\\u0000")
    inv.sense_id = inv.sense_id.astype(str)
    inv.cluster = inv.cluster.astype(str)
    
    for _, row in inv.iterrows():
        word = row.word
        size = len(row.cluster.split(','))
        
        if word in mapping:
            mf_sense, max_size = mapping[row.word]
            if size > max_size:
                mapping[row.word] = (row.sense_id, size)
        else:
            mapping[row.word] = (row.sense_id, size)
            
    return mapping


def run(test_file, output, mapping):
    print("Loading test set...")
    reader = read_csv(test_file, encoding="utf-8", delimiter="\t", dtype={'predict_related': object, 'gold_sense_ids':object, 'predict_sense_ids':object})
    rows_count = reader.shape[0]
    print(rows_count, " test instances")
    
    for i, row in reader.iterrows():
        if row.target in mapping:
            reader.set_value(i, 'predict_sense_ids', mapping[row.target][0])
    
    reader.to_csv(sep='\t', path_or_buf=output, encoding="utf-8", index=False, quoting=QUOTE_NONE)
    print("Saved predictions to ", output)
    

def main():
    parser = argparse.ArgumentParser(description='Fill in a test dataset using Most Frequent Sense method.')
    parser.add_argument('test_file', help='A path to a test dataset. Format: "context_id<TAB>target<TAB>target_pos<TAB>target_position<TAB>gold_sense_ids<TAB>predict_sense_ids<TAB>golden_related<TAB>predict_related<TAB>context')
    parser.add_argument("inventory", help="A path to a sense inventory")
    parser.add_argument("output", help="An output path to the filled dataset. Same format as test_file")
    
    args = parser.parse_args()
    
    mapping = mfs_mapping(args.inventory)

    run(args.test_file, args.output, mapping) 


if __name__ == '__main__':
    main()