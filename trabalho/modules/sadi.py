import random
import math
import numpy as np
import networkx as nx
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# k-means com Dissimilarity Index
# ---------------------------------------------------------------------------

def assign_communities(Delta, centers):
    """
    Atribui cada nó à comunidade cujo centro tem menor dissimilaridade.
    S_k = { x : k = argmin_l Δ(x, center_l) }   [Equação 13]
    """
    labels = np.argmin(Delta[:, centers], axis=1)
    communities = {}
    for k, c in enumerate(centers):
        members = np.where(labels == k)[0].tolist()
        communities[k] = members if members else [c]
    return communities, labels


def compute_center(community_nodes, Delta):
    """
    m(S_k) = argmin_{x in S_k} mean_{y in S_k, y≠x} Δ(x, y)   [Equação 5]
    """
    if len(community_nodes) == 1:
        return community_nodes[0]
    best_node, best_score = None, np.inf
    for x in community_nodes:
        others = [y for y in community_nodes if y != x]
        avg = np.mean(Delta[x, others])
        if avg < best_score:
            best_score, best_node = avg, x
    return best_node


def community_strength(G, community_nodes):
    """
    M(S_k) = sum_{x in S_k} (d_in(x) - d_out(x))   [Equação 15]
    """
    node_set = set(community_nodes)
    strength = 0
    for x in community_nodes:
        d_in  = sum(1 for nb in G.neighbors(x) if nb in node_set)
        d_out = sum(1 for nb in G.neighbors(x) if nb not in node_set)
        strength += d_in - d_out
    return strength


# ---------------------------------------------------------------------------
# Funções objetivo
# ---------------------------------------------------------------------------

def compute_silhouette(Delta, labels):
    """Silhouette Score usando Δ como métrica de distância (metric='precomputed')."""
    if len(np.unique(labels)) < 2:
        return -1.0
    try:
        return silhouette_score(Delta, labels, metric='precomputed')
    except Exception:
        return -1.0


def compute_modularity(G, labels):
    """Modularity Q de Newman-Girvan via NetworkX."""
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    communities_sets = [set(np.where(labels == k)[0]) for k in unique]
    communities_sets = [c for c in communities_sets if c]
    try:
        return nx.community.modularity(G, communities_sets)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Funções de resfriamento (cooling schedules)
# ---------------------------------------------------------------------------

def _make_cooling_fn(cooling, T_max, T_min, alpha):
    """
    Retorna uma função  schedule(step: int) -> float  que mapeia o índice do
    passo de temperatura (0-based) para o valor de T.

    Esquemas disponíveis:
        'geometric'    T_k = T_max * alpha^k
        'linear'       T_k = T_max - k * (T_max - T_min) / n_steps
        'logarithmic'  T_k = T_max / log(1 + k + e)   (normalizado para
                       coincidir com T_min no mesmo n_steps da geométrica)

    Também aceita um callable  f(step, T_max, T_min, alpha) -> float.
    """
    # Número de passos equivalente ao esquema geométrico (usado para calibrar
    # linear e logarítmica para que terminem aproximadamente no mesmo ponto).
    n_steps = math.ceil(math.log(T_min / T_max) / math.log(alpha))

    if callable(cooling):
        return lambda step: cooling(step, T_max, T_min, alpha)

    if cooling == 'geometric':
        return lambda step: T_max * (alpha ** step)

    if cooling == 'linear':
        step_size = (T_max - T_min) / n_steps
        return lambda step: max(T_min, T_max - step * step_size)

    if cooling == 'logarithmic':
        # Escala C tal que schedule(n_steps) ≈ T_min
        # T_min = C / log(1 + n_steps + e)  =>  C = T_min * log(1 + n_steps + e)
        C = T_min * math.log(1 + n_steps + math.e)
        return lambda step: max(T_min, C / math.log(1 + step + math.e))

    raise ValueError(f"cooling desconhecido: {cooling!r}. "
                     "Use 'geometric', 'linear', 'logarithmic' ou um callable.")


# ---------------------------------------------------------------------------
# Algoritmo SADI completo
# ---------------------------------------------------------------------------

def sadi_algorithm(G, Delta,
                   T_max=3.0, T_min=0.01,
                   alpha=0.9, R=20,
                   N_min=2, N_max=None,
                   cooling='geometric',
                   objective='silhouette'):
    """
    Simulated Annealing com Dissimilarity Index k-means (SADI).

    Parâmetros:
        T_max, T_min : temperaturas inicial e final
        alpha        : taxa de resfriamento (0 < alpha < 1)
        R            : iterações por temperatura
        N_min, N_max : limites para número de comunidades
        cooling      : 'geometric' (padrão), 'linear', 'logarithmic' ou callable
        objective    : 'silhouette' (padrão) ou 'modularity'

    Retorna (best_labels, best_centers, best_score, history).
    history: lista de tuplas (iteration, temperature, current_score, best_score, n_comm).
    """
    n = len(G.nodes())
    if N_max is None:
        N_max = max(2, n // 3)

    if objective == 'modularity':
        score_fn = lambda lbl: compute_modularity(G, lbl)
    else:
        score_fn = lambda lbl: compute_silhouette(Delta, lbl)

    nodes = list(range(n))

    centers = random.sample(nodes, random.randint(N_min, N_max))
    communities, labels = assign_communities(Delta, centers)
    energy = -score_fn(labels)

    best_energy, best_labels, best_centers = energy, labels.copy(), centers.copy()

    schedule = _make_cooling_fn(cooling, T_max, T_min, alpha)
    history = []
    iteration = 0
    temp_step = 0
    T = schedule(temp_step)

    while T > T_min:
        for _ in range(R):
            op = random.choice(['retain', 'delete', 'split'])
            new_centers = centers.copy()

            if op == 'delete' and len(new_centers) > N_min:
                strengths = {k: community_strength(G, communities[k]) for k in communities}
                weakest = min(strengths, key=strengths.get)
                if weakest < len(new_centers):
                    new_centers.pop(weakest)

            elif op == 'split' and len(new_centers) < N_max:
                avg_strengths = {
                    k: community_strength(G, communities[k]) / len(communities[k])
                    for k in communities if len(communities[k]) > 1
                }
                if avg_strengths:
                    split_k = min(avg_strengths, key=avg_strengths.get)
                    split_center = new_centers[split_k]
                    others = [x for x in communities[split_k] if x != split_center]
                    if others:
                        new_c = max(others, key=lambda x: Delta[x, split_center])
                        new_centers.append(new_c)

            new_comm, new_labels = assign_communities(Delta, new_centers)
            new_centers = [compute_center(new_comm[k], Delta) for k in range(len(new_centers))]
            new_comm, new_labels = assign_communities(Delta, new_centers)

            new_energy = -score_fn(new_labels)
            delta_E = new_energy - energy

            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                centers, communities, labels, energy = new_centers, new_comm, new_labels, new_energy
                if energy < best_energy:
                    best_energy, best_labels, best_centers = energy, labels.copy(), centers.copy()

            history.append((iteration, T, -energy, -best_energy, len(centers)))
            iteration += 1

        temp_step += 1
        T = schedule(temp_step)

    return best_labels, best_centers, -best_energy, history
