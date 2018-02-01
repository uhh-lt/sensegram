from networkx_server import NetworxServerManager

neighbors_fpath = "model/text8.graph"
G = NetworxServerManager(neighbors_fpath)

while True:
    i = input()
    if i == "q": break
