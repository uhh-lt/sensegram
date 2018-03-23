from chinese_whispers import chinese_whispers, aggregate_clusters
from networkx import Graph
from multiprocessing import Pool
import codecs
from time import time
from graph import CRSGraph, WEIGHT_COEF
import re 


G = None
n = None
verbose = True


first_not_letter = re.compile(r"^[^a-z]",re.IGNORECASE)
last_not_letter = re.compile(r"[^a-z]$", re.IGNORECASE)

def minimize(word):
    word = word.strip().lower()
    
    if first_not_letter.findall(word):
        word = word[1:]

    if last_not_letter.findall(word):        
        word = word[:-1]
        
    return word


def get_ego_network(ego):
    tic = time()
    ego_network = Graph(name=ego)
    
    # Add related and substring nodes
    substring_nodes = []
    for j, node in enumerate(G.index):
        if node == ego:

            ego_nn_nodes = []
            for related_node, related_weight in G.get_neighbors(node).items():
                if minimize(related_node) == minimize(ego): continue
                ego_nn_nodes.append( (related_node, {"weight": related_weight}) )

            ego_network.add_nodes_from(ego_nn_nodes)
        else:
            if "_" not in node: continue
            if node.startswith(ego + "_") or node.endswith("_" + ego):
                substring_nodes.append( (node, {"weight": WEIGHT_COEF}) )
    ego_network.add_nodes_from(substring_nodes)
    
    # Find edges of the ego network
    for r_node in ego_network:
        related_related_nodes = G.get_neighbors(r_node)
        related_related_nodes_ego = sorted(
            [(related_related_nodes[rr_node], rr_node) for rr_node in related_related_nodes if rr_node in ego_network],
            reverse=True)[:n]
        related_edges = [(r_node, rr_node, {"weight": w}) for w, rr_node in  related_related_nodes_ego]
        ego_network.add_edges_from(related_edges)
    
    chinese_whispers(ego_network, weighting="top", iterations=20)
    if verbose: print("{}\t{:f} sec.".format(ego, time()-tic))

    return ego_network


def ego_network_clustering(neighbors_fpath, clusters_fpath, max_related=300, num_cores=32):
    global G
    global n
    G = CRSGraph(neighbors_fpath)
    
    with codecs.open(clusters_fpath, "w", "utf-8") as output, Pool(num_cores) as pool:
        output.write("word\tcid\tcluster\tisas\n")

        for i, ego_network in enumerate(pool.imap_unordered(get_ego_network, G.index)):
            if i % 1000 == 0: print(i, "ego networks processed")
            sense_num = 1
            for label, cluster in sorted(aggregate_clusters(ego_network).items(), key=lambda e: len(e[1]), reverse=True):
                output.write("{}\t{}\t{}\t\n".format(
                    ego_network.name,
                    sense_num,
                    ", ".join(
                        ["{}:{:.4f}".format(n,w) for w, n in sorted([(ego_network.node[c_node]["weight"]/WEIGHT_COEF, c_node) for c_node in cluster], reverse=True)]
                        )))
                sense_num += 1
    print("Clusters:", clusters_fpath)
