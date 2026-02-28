import numpy as np
import networkx as nx


def correlation_network(returns_matrix, threshold=0.5):

    corr = np.corrcoef(returns_matrix.T)
    N = corr.shape[0]

    G = nx.Graph()

    for i in range(N):
        for j in range(i+1, N):
            if abs(corr[i, j]) > threshold:
                G.add_edge(i, j, weight=corr[i, j])

    return G


def network_stress_index(G):

    if len(G.nodes) == 0:
        return 0.0

    density = nx.density(G)

    try:
        largest_cc = len(max(nx.connected_components(G), key=len))
    except ValueError:
        largest_cc = 0

    clustering = nx.average_clustering(G)

    return density + clustering + (largest_cc / max(1, len(G.nodes)))