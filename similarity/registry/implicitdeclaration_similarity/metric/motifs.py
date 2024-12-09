import copy
import queue
import networkx as nx
import time
import matplotlib.pyplot as plt
import numpy
from numpy import sort


class type_counter:
    def __init__(self):
        self.type1 = 0  # 3 nodes of the same category
        self.type2 = 0  # 2 nodes of the same category, 1 node of the different category
        self.type3 = 0  # 3 nodes of the different category

    def __repr__(self):
        print("Type1 motif:", self.type1)
        print("Type2 motif:", self.type2)
        print("Type3 motif:", self.type3)
        return "Motif counter"

class motif:
    def __init__(self):
        self.v_subgraph = []  # store the nodes of motifs
        self.edge_list = []   # store the sets of edges
        self.node_label = []  # store the labels of nodes

    def __repr__(self):
        print("nodes:", self.v_subgraph)
        print("labels:", self.node_label)
        print("edges:", self.edge_list)
        return "Motif Type1"  # Triangle motif


def ESU_BFS(undirected_adj, target):
    '''

    Args:
        undirected_adj: undirected adjacency matrix
        target: labels of nodes

    Returns: None

    '''
    Type_counter = type_counter()
    undirected_unweighted_adj = (undirected_adj != 0).astype(int)
    # print(undirected_unweighted_adj)
    G = nx.from_numpy_matrix(undirected_unweighted_adj)
    # pos = nx.circular_layout(G)  # for visualize
    # nx.draw(G, with_labels=True, pos=pos)
    # plt.savefig('undirecetd_weighted_network.jpg')
    # plt.show()
    start = time.time()
    for node in G.nodes:
        get_motifs(G, undirected_unweighted_adj, target, node, list(G.nodes)[-1], Type_counter)
    end = time.time()
    # print('Time:', end-start)
    # print(Type_counter)
    return Type_counter


def get_motifs(G, undirected_unweighted_adj, target, root, max_node, Type_counter):
    '''

    Args:
        G: target graph
        undirected_unweighted_adj: adjacency matrix of the target graph
        target: labels of nodes
        root: the root node
        max_node: Maximum node number of the target graph
        Type_counter: count the number of different motifs

    Returns:

    '''
    st = [0 for i in range(len(G.nodes))]  # Initialize nodes state
    st[root] = 1  # Mark the current node
    q = queue.Queue()
    m = motif()
    m.v_subgraph.append(root)
    m.node_label.append(target[root])
    q.put(m)
    while (not q.empty()):
        u = q.get()  # 取出
        if (len(u.v_subgraph) == 3):
            type = classify(u, undirected_unweighted_adj, Type_counter)
            # if type is not None:
            #     print(type)
            continue
        node = u.v_subgraph[-1]  # root
        st[node] = 1
        for i in range(node + 1, max_node + 1):
            if st[i] == 1: continue
            if G.has_edge(node, i):
                new_motif = copy.deepcopy(u)
                new_motif.v_subgraph.append(i)
                new_motif.node_label.append(target[i])
                q.put(new_motif)


def classify(u, adj, Type_counter):
    '''
    Classify the obtained third-order motif, if there are two edges, it is type1, if there are three edges, it is type2.
    Args:
        u: motif needs to be classified
        adj: adjacency matrix of the target graph
        Type_counter: count the number of different motifs
    Returns: motif

    '''
    type1 = []  # Triangle motif
    if adj[u.v_subgraph[0]][u.v_subgraph[1]] == 1 and adj[u.v_subgraph[1]][u.v_subgraph[0]] == 1:
        u.edge_list.append((u.v_subgraph[0], u.v_subgraph[1]))
    if adj[u.v_subgraph[0]][u.v_subgraph[2]] == 1 and adj[u.v_subgraph[2]][u.v_subgraph[0]] == 1:
        u.edge_list.append((u.v_subgraph[0], u.v_subgraph[2]))
    if adj[u.v_subgraph[1]][u.v_subgraph[2]] == 1 and adj[u.v_subgraph[2]][u.v_subgraph[1]] == 1:
        u.edge_list.append((u.v_subgraph[1], u.v_subgraph[2]))

    if len(u.edge_list) == 3:
        type1.append(u)
        b = sort(u.node_label)
        if b[0] == b[1] and b[1] == b[2]:
            Type_counter.type1 += 1
        elif b[0] != b[1] and b[1] == b[2]:
            Type_counter.type2 += 1
        elif b[0] == b[1] and b[1] != b[2]:
            Type_counter.type2 += 1
        elif b[0] != b[1] and b[1] != b[2] and b[0] != b[2]:
            Type_counter.type3 += 1
        return type1



if __name__ == "__main__":
    input = numpy.array([[0,1,0,0,0,1,0,1],
                         [1,0,1,1,1,0,1,1],
                         [0,1,0,1,0,1,0,0],
                         [0,1,1,0,1,0,1,0],
                         [0,1,0,1,0,1,1,1],
                         [1,0,1,0,1,1,0,1],
                         [0,1,0,1,1,0,0,1],
                         [1,1,0,0,1,1,1,0]])
    labels = [1, 2, 1, 1, 2, 1, 2, 1]
    ESU_BFS(input, labels)

