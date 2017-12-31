import argparse
from nltk.parse.stanford import StanfordDependencyParser
import os
from pandas import read_csv, Series
from csv import QUOTE_NONE
from spacy.en import English


_spacy = English()


os.environ["JAVA_HOME"] = "/home/fahrer/jdk1.8.0_45/"

path_to_jar = '/home/pelevina/stanford-parser-full-2015-12-09/stanford-parser.jar'
path_to_models_jar = '/home/pelevina/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar, java_options='-Xmx4096m')

def lemmatize(text):
    tokens = _spacy(text, tag=True, parse=False, entity=True)
    text_lemmatized = " ".join(t.lemma_ for t in tokens)
    return text_lemmatized

def run(dataset, block):
    
    print("Reading the dataset.")
    reader = read_csv(dataset, encoding="utf-8", delimiter="\t", dtype={'predict_related': object, 'gold_sense_ids':object, 'predict_sense_ids':object})
    rows_count = reader.shape[0]
    lst = ["initDep"] * rows_count
    reader['deps'] = Series(lst, index=reader.index)
    contexts = reader['context'].tolist()
    contexts = [c.lower() for c in contexts]
    
    print("Parsing sentences.")
    for j in range(0, rows_count, block):
        print(("j =", j))
        result = dependency_parser.raw_parse_sents(contexts[j:j+block])
        result = list(result)
        for i, parse in enumerate(result):
            if parse == "ParseError":
                reader.set_value(j + i, 'deps', parse)
                print(("Parse error at index = ", i))
                print((contexts[j + i]))
            else:
                deplist = list(parse.next().triples())

                target_position = reader["target_position"].iloc[j + i]
                start, end = [int(x) for x in target_position.split(',')]
                word = contexts[j + i][start:end]
                
                target = reader["target"].iloc[j + i]

                ctx1 = [rel + "_" + tail[0] for head, rel, tail in deplist if lemmatize(head[0]) == target or head[0] == word]
                ctx2 = [rel + "I_" + head[0] for head, rel, tail in deplist if lemmatize(tail[0]) == target or tail[0] == word]
                reader.set_value(j + i, 'deps', " ".join(ctx1 + ctx2))
    
    print(("Saving the dataset to ", dataset+".dep.csv"))
    reader.to_csv(sep='\t', path_or_buf = dataset + '.dep.csv', encoding="utf-8", index=False, quoting=QUOTE_NONE)

    
    
def main():
    parser = argparse.ArgumentParser(description='Add dependencies to TWSI dataset')
    parser.add_argument('dataset', help='A path to a test dataset. Format: "context_id<TAB>target<TAB>target_pos<TAB>target_position<TAB>gold_sense_ids<TAB>predict_sense_ids<TAB>golden_related<TAB>predict_related<TAB>context')
    parser.add_argument("-block", help="Number of sentences to parse as one block ", default=5000, type=int)
    args = parser.parse_args()

    run(args.dataset, args.block) 

if __name__ == '__main__':
    main()
