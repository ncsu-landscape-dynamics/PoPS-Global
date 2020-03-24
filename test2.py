import networkx as nx
G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2,3])

G.add_edge(1,2 , imports = 7, exports = 4,)
G.add_edge(1, 