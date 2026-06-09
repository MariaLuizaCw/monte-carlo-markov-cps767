import numpy as np
import networkx as nx
from scipy.linalg import solve


def build_markov_chain(G):
    """
    Constrói a matriz de transição P a partir do grafo G.
    p(x, y) = w(x,y) / d(x)   [Equação 1 do artigo]
    Retorna (P, nodes, node_idx).
    """
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, weight='weight')
    degrees = A.sum(axis=1)
    degrees[degrees == 0] = 1
    P = A / degrees[:, np.newaxis]
    node_idx = {node: i for i, node in enumerate(nodes)}
    return P, nodes, node_idx


def compute_mfpt_matrix(P):
    """
    Calcula a matriz de mean first-passage times T[x, y].
    Para cada y resolve: (I - B(y)) * t(:,y) = 1, onde B(y) é P com coluna y zerada.
    [Equações 2-3 do artigo] — custo O(n³).
    """
    n = P.shape[0]
    T = np.zeros((n, n))
    I = np.eye(n)

    for y in range(n):
        B = P.copy()
        B[:, y] = 0.0
        A_sys = I - B
        try:
            T[:, y] = solve(A_sys, np.ones(n))
        except np.linalg.LinAlgError:
            T[:, y] = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)[0]

    return T


def compute_dissimilarity_matrix(T):
    """
    Δ(x, y) = sqrt( 1/(n-2) * sum_{z ≠ x,y} (T[x,z] - T[y,z])^2 )
    [Equação 4 do artigo]
    """
    n = T.shape[0]
    Delta = np.zeros((n, n))

    for x in range(n):
        for y in range(x + 1, n):
            mask = np.ones(n, dtype=bool)
            mask[x] = False
            mask[y] = False
            diff = T[x, mask] - T[y, mask]
            d = np.sqrt(np.sum(diff ** 2) / (n - 2))
            Delta[x, y] = d
            Delta[y, x] = d

    return Delta


def build_dissimilarity(G):
    """Atalho: executa os três passos em sequência e retorna Delta."""
    P, nodes, node_idx = build_markov_chain(G)
    T = compute_mfpt_matrix(P)
    Delta = compute_dissimilarity_matrix(T)
    return Delta, nodes, node_idx
