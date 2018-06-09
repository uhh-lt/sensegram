import argparse
from word_sense_induction import minimize 
from chinese_whispers import chinese_whispers, aggregate_clusters
from networkx import Graph
from gensim.models import KeyedVectors
from time import time 
import networkx as nx
import matplotlib.pyplot as plt
from pandas import read_csv
from glob import glob 
from collections import Counter
import codecs
from traceback import format_exc
import gzip
import logging
import gensim
import requests
from clint.textui import progress
from os.path import join, exists


wsi_data_dir = "/home/panchenko/russe-wsi-full/data/"


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TOPN = 50
verbose = True

try:
    wv
except NameError:
    wv = None


def get_ru_wsi_vocabulary():
    dataset_names = ["active-dict", "wiki-wiki", "bts-rnc"]

    voc = set(["ключ", "замок", "коса"])

    for dataset_name in dataset_names:
        train_fpath = join(join(wsi_data_dir, dataset_name), "train.csv")
        test_fpath = join(join(wsi_data_dir, dataset_name), "test.csv")

        if exists(train_fpath):
            train = read_csv(train_fpath, sep="\t", encoding="utf-8")

        if exists(test_fpath):
            test = read_csv(test_fpath, sep="\t", encoding="utf-8")

        for i, row in test.iterrows(): voc.add(row.word)
        for i, row in train.iterrows(): voc.add(row.word)

    return voc


def get_sorted_vocabulary(vectors_fpath):
    with gzip.open(vectors_fpath, "rb") as in_f:
        vocabulary = []
        for i, line in enumerate(in_f):
            if i == 0: continue
            vocabulary.append( str(line, "utf-8").split(" ")[0] )
    return vocabulary
            
def save_to_gensim_format(wv, output_fpath):
    tic = time()
    wv.save(output_fpath)
    print("Saved in {} sec.".format(time()-tic))
    

def load_globally(word_vectors_fpath):
    global wv
    
    if not wv:
        print("Loading word vectors from:", word_vectors_fpath)
        tic = time()
        if word_vectors_fpath.endswith(".vec.gz"):
            wv = KeyedVectors.load_word2vec_format(word_vectors_fpath, binary=False, unicode_errors="ignore")
        else:
            wv = gensim.models.KeyedVectors.load(word_vectors_fpath)            
        print("Loaded in {} sec.".format(time()-tic))
    else:
        print("Using loaded word vectors.")

    return wv


def get_nns(target, topn=TOPN):
    nns = wv.most_similar(positive=[target], negative=[], topn=topn)
    nns = [(word, score) for word, score in nns if minimize(word) != minimize(target)]
    return nns


def in_nns(nns, word):
    for w, s in nns:
        if minimize(word) == minimize(w):
            return True
        
    return False 


def get_pair(first, second):
    pair_lst = sorted([first, second])
    sorted_pair = (pair_lst[0], pair_lst[1])
    return sorted_pair         


def get_disc_pairs(ego, topn=TOPN):  
    pairs = set()
    nns = get_nns(ego, topn)
    
    for i in range(len(nns)):
        topi = nns[i][0]
        nns_topi = get_nns(topi, topn) 
        nns_untopi = wv.most_similar(positive=[ego], negative=[topi], topn=topn)
        untopi = nns_untopi[0][0]
        if in_nns(nns, untopi): pairs.add(get_pair(topi, untopi))

    return pairs


def get_nodes(pairs):
    nodes = Counter()
    for src, dst in pairs:
        nodes.update([src])
        nodes.update([dst])
        
    return nodes


def list2dict(lst):
    return {p[0]: p[1] for p in lst}


def wsi(ego, topn=TOPN):
    tic = time()
    ego_network = Graph(name=ego)

    pairs = get_disc_pairs(ego, topn)
    nodes = get_nodes(pairs)   
    
    ego_network.add_nodes_from( [(node, {'size': size}) for node, size in nodes.items()] )
    
    for r_node in ego_network:
        related_related_nodes = list2dict(get_nns(r_node))
        related_related_nodes_ego = sorted(
            [(related_related_nodes[rr_node], rr_node) for rr_node in related_related_nodes if rr_node in ego_network],
            reverse=True)[:topn]
        
        related_edges = []
        for w, rr_node in related_related_nodes_ego:
            if get_pair(r_node, rr_node) not in pairs:
                related_edges.append( (r_node, rr_node, {"weight": w}) )
            else:
                print("Skipping:", r_node, rr_node)
        ego_network.add_edges_from(related_edges)

    chinese_whispers(ego_network, weighting="top", iterations=20)
    if verbose: print("{}\t{:f} sec.".format(ego, time()-tic))

    return {"network": ego_network,  "nodes": nodes}


def draw_ego(G, show=False, save_fpath=""):
    colors = [1. / G.node[node]['label'] for node in G.nodes()]
    sizes = [300. * G.node[node]['size'] for node in G.nodes()]  

    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(20, 20)

    nx.draw_networkx(G, cmap=plt.get_cmap('gist_rainbow'),
                     node_color=colors,
                     font_color='black',
                     node_size=sizes)

    if show: plt.show()
    if save_fpath != "":
        plt.savefig(save_fpath)
        
    fig.clf()
        

def get_target_words(language):
    """ Takes as input a two symbol language code e.g. 'de' and returns all 
    words from the evaluation datasets for this language """ 

    words = set()

    for pairs_fpath in glob("eval/data/{}*dataset".format(language)):
        df = read_csv(pairs_fpath, sep=";", encoding="utf-8")
        for i, row in df.iterrows():
            words.add(row.word1)
            words.add(row.word2)

    words = sorted(words)
    return words


def get_cluster_lines(G, nodes):
    lines = []
    labels_clusters = sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True)
    for label, cluster in labels_clusters:
        scored_words = []
        for word in cluster:
            scored_words.append( (nodes[word], word) )
        keyword = sorted(scored_words, reverse=True)[0][1]
        
        lines.append("{}\t{}\t{}\t{}\n".format(G.name, label, keyword, ", ".join(cluster)))
        
    return lines 


def run(language="ru", eval_vocabulary=False, visualize=True, show_plot=False):
    # parameters
    wv_fpath = "model/cc.{}.300.vec.gz".format(language)
    wv_pkl_fpath = wv_fpath + ".pkl"

    if eval_vocabulary:
        voc = get_target_words(language)    
    else:
        voc = get_sorted_vocabulary(wv_fpath)
    words = {w: None for w in voc}    words = {w: None for w in voc}

    print("Language:", language)
    print("Visuzlize:", visualize)
    print("Vocabulary: {} words", len(voc))

    # ensure that the word vectors exist
    if not exists(wv_fpath):
	wv_uri = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.{}.300.vec.gz".format(language)
	print("Downloading the fasttext model from {}".format(wv_uri))
	r = requests.get(wv_uri, stream=True)
	path = "model/cc.{}.300.vec.gz".format(language)
	with open(path, "wb") as f:
	    total_length = int(r.headers.get("content-length"))
	    for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
		if chunk:
		    f.write(chunk)
		    f.flush()

    # ensure the word vectors are saved in the fast to load gensim format 
    if not exists(wv_pkl_fpath):
	load_globally(wv_fpath) # loads wv 
	save_to_gensim_format(wv, wv_pkl_fpath)
    else:
	load_globally(wv_pkl_fpath)

    # perform word sense induction 
    for topn in [50, 100, 200]: 
	output_fpath = wv_fpath + ".top{}.wsi-inventory.tsv".format(topn)
	with codecs.open(output_fpath, "w", "utf-8") as out:
	    out.write("word\tcid\tkeyword\tcluster\n")
	    for word in words:
		try:
		    words[word] = wsi(word, topn=topn)
		    if visualize:
			plt_fpath = output_fpath + ".{}.png".format(word)
			draw_ego(words[word]["network"], show_plot, plt_fpath)
		    lines = get_cluster_lines(words[word]["network"], words[word]["nodes"])
		    for l in lines: out.write(l)
		except KeyboardInterrupt:
		    break
		except:
		    print("Error:", word)
		    print(format_exc())
	print("Output:", output_fpath)

def run_old(language = "de", eval_vocabulary=False, visualize = False, show_plot = False):
    print("Language:", language)
    print("Evaluation:", eval_vocabulary)
    print("Visuzlize:", visualize)

    wv_fpath = "model/cc.{}.300.vec.gz".format(language)
    wv_gensim_fpath = wv_fpath + ".gensim"

    if not exists(wv_fpath):
        wv_uri = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.{}.300.vec.gz".format(language)
        print("Downloading the fasttext model from {}".format(wv_uri))
        r = requests.get(wv_uri, stream=True)
        path = "model/cc.{}.300.vec.gz".format(language)
        with open(path, "wb") as f:
            total_length = int(r.headers.get("content-length"))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                if chunk:
                    f.write(chunk)
                    f.flush()
                    
    if eval_vocabulary:
        voc = get_target_words(language)    
    else:
        voc = get_sorted_vocabulary(wv_fpath)
    words = {w: None for w in voc}

    if not exists(wv_gensim_fpath):
        load_globally(wv_fpath) # loads wv 
        save_to_gensim_format(wv, wv_gensim_fpath)
    else:
        load_globally(wv_gensim_fpath)
        
    for topn in [50, 100, 200]: 
        output_fpath = wv_fpath + ".top{}.inventory.tsv".format(topn)
        with codecs.open(output_fpath, "w", "utf-8") as out:
            out.write("word\tcid\tkeyword\tcluster\n")
            for word in words:
                try:
                    words[word] = wsi(word, topn=topn)
                    if visualize:
                        plt_fpath = output_fpath + ".{}.png".format(word)
                        draw_ego(words[word]["network"], show_plot, plt_fpath)
                    lines = get_cluster_lines(words[word]["network"], words[word]["nodes"])
                    for l in lines: out.write(l)
                except KeyboardInterrupt:
                    break
                except:
                    print("Error:", word)
                    print(format_exc())
        print("Output:", output_fpath)
    

def main():
    parser = argparse.ArgumentParser(description='Graph-Vector Word Sense Induction appraoch.')
    parser.add_argument("language", help="A code that represents input language, e.g. 'en', 'de' or 'ru'. ")
    parser.add_argument("-eval", help="Use only evaluation vocabulary, not all words in the model.", action="store_true")
    parser.add_argument("-viz", help="Visualize each ego networks.", action="store_true")
    args = parser.parse_args()

    run(language=args.language, eval_vocabulary=args.eval, visualize=args.viz) 


if __name__ == '__main__':
    main()
