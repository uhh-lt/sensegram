from multiprocessing import Manager, Process
import Pyro4
import networkx as nx 
from time import time, sleep


@Pyro4.expose
class NetworkxServer(object):
    def __init__(self, neighbors_fpath):
        tic = time()
        print("Loading the graph:", neighbors_fpath)
        self.G = nx.read_edgelist(
           neighbors_fpath,
           nodetype=str,
           delimiter="\t",
           data=(('weight',float),))
        
        print("\nLoaded in {:f} sec.".format(time() - tic))
                   
    def get_neighbors(self, node):
        try:
            return self.G[node]
        except:
            print("Error:", node)
            return []
    
    def get_node(self, node):
        try:
            return self.G.nodes[node]
        except:
            print("Error:", node)
            return {}

    def get_edge(self, node_i, node_j):
        try:
            return self.G[node_i][node_j]    
        except:
            print("Error:", node_i, node_j)
            return {}

    def get_node_list(self):
        return self.G.nodes.keys()
    

def run_pyro_daemon(neighbors_fpath):
    networkx_server = NetworkxServer(neighbors_fpath)
    daemon = Pyro4.Daemon()   
    uri = daemon.register(networkx_server)
    d["uri"] = uri
    daemon.requestLoop()
    

d = Manager().dict()


class NetworxServerManager(object):
    def __init__(self, neighbors_fpath):
        
        self._graph_server = self._graph_server = Process(
                name='NetworkX RPC server',
                target=run_pyro_daemon,
                args=(neighbors_fpath,))
        self._graph_server.daemon = True
        self._graph_server.start()
        
        global d
        d["uri"] = ""
        while(d["uri"] == ""):
            sleep(1)
            print(".", end="")
            
        self._uri = d["uri"]
        print("\nURI:", self._uri)        
        self.graph = Pyro4.Proxy(self._uri) 
        
    def stop(self):
        self._graph_server.terminate()        
    
# sample usage:
# neighbors_fpath = "model/text8.graph"
# s = NetworxServerManager(neighbors_fpath)
# s.graph...
