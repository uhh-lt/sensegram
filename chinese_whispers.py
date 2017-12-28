import sys
import subprocess
from pandas import read_csv


def sort_input_sort(input_fpath):
    sorted_fpath = input_fpath + ".sort"
    df = read_csv(input_fpath, sep="\t", encoding='utf8', names=["node_i", "node_j", "sim_ij"])
    df = df.sort_values(["node_i", "sim_ij"], ascending=False)
    df.to_csv(sorted_fpath, sep="\t", header=False, index=False, encoding="utf-8")
    return sorted_fpath


def clustering(input_graph_fpath, output_clusters_fpath="", mode="global", clustering="cw", N=200, n=200, cw_option="TOP", java_xmx_gb=16):
    """
    input_graph_fpath in the 'node_i<TAB>node_j<TAB>sim_ij' format. Can be sorted in any way. 
    mode: wsi or global; clustering: cw or mcl """

    sorted_input_fpath = sort_input_sort(input_graph_fpath)

    if output_clusters_fpath == "": output_clusters_fpath = input_graph_fpath + "-%s-%s-%d-%d-%s.clusters" % (mode, clustering, N, n, cw_option)

    if mode == "wsi":
        cmd = "java -Xms1G -Xmx%dG -cp chinese-whispers/target/chinese-whispers.jar de.tudarmstadt.lt.wsi.WSI -in %s -out %s -N %d -n %d  -clustering %s -cwOption %s" % (
            java_xmx_gb, sorted_input_fpath, output_clusters_fpath, N, n, clustering, cw_option)
    else:
        cmd = "java -Xms1G -Xmx%dG -cp chinese-whispers/target/chinese-whispers.jar de.tudarmstadt.lt.cw.global.CWGlobal -in %s -out %s -N %d -cwOption %s" % (
            java_xmx_gb, sorted_input_fpath, output_clusters_fpath, N, cw_option)

    print "\nStart clustering of with the following parameters:"
    print cmd

    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)

    print "Output clusters:", output_clusters_fpath
    print "Sorted input graph:", sorted_input_fpath


