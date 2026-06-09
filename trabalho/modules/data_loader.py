import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_rand_score


def load_karate_club():
    """Retorna (G, true_labels). Ground truth: Mr. Hi=0, Officer=1."""
    raw = nx.karate_club_graph()
    club_map = {'Mr. Hi': 0, 'Officer': 1}
    raw_labels = {v: club_map[raw.nodes[v]['club']] for v in raw.nodes()}
    G, orig_nodes = _prepare(raw)
    true_labels = [raw_labels[v] for v in orig_nodes]
    return G, true_labels


def _read_gml_from_zip(path):
    """Lê GML de dentro de um zip ou diretamente de um arquivo .gml."""
    import zipfile
    if path.endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            gml_name = next(n for n in z.namelist() if n.endswith('.gml'))
            with z.open(gml_name) as f:
                content = f.read().decode('utf-8', errors='replace')
        return nx.parse_gml(content)
    return nx.read_gml(path)


def load_dolphins(path):
    """Retorna (G, None). Sem ground truth oficial."""
    raw = _read_gml_from_zip(path)
    G, _ = _prepare(raw)
    return G, None


def load_lesmis(path):
    """Retorna (G, None). Grafo ponderado de Les Misérables. Sem ground truth."""
    raw = _read_gml_from_zip(path)
    G, _ = _prepare(raw)
    return G, None


def load_football(path):
    """Retorna (G, true_labels). Ground truth: atributo 'value' (conferência 0-11)."""
    raw = _read_gml_from_zip(path)
    raw_labels = {v: raw.nodes[v]['value'] for v in raw.nodes()}
    G, orig_nodes = _prepare(raw)
    true_labels = [raw_labels[v] for v in orig_nodes]
    return G, true_labels


def load_polbooks(path):
    """Retorna (G, true_labels). Ground truth: atributo 'value' mapeado para 0=liberal, 1=neutro, 2=conservador."""
    raw = _read_gml_from_zip(path)
    raw_vals = {v: raw.nodes[v]['value'] for v in raw.nodes()}
    unique = sorted(set(raw_vals.values()))
    remap = {orig: new for new, orig in enumerate(unique)}
    raw_labels = {v: remap[raw_vals[v]] for v in raw_vals}
    G, orig_nodes = _prepare(raw)
    true_labels = [raw_labels[v] for v in orig_nodes]
    return G, true_labels


def load_polblogs(path):
    """
    Retorna (G, true_labels). Ground truth: atributo 'value' (0=liberal, 1=conservador).
    Lê o GML de dentro de um zip ou de um arquivo direto.
    """
    import zipfile, io
    if path.endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            gml_name = next(n for n in z.namelist() if n.endswith('.gml'))
            with z.open(gml_name) as f:
                content = f.read().decode('utf-8')
        raw = nx.parse_gml(content)
    else:
        raw = nx.read_gml(path)
    raw = raw.to_undirected()
    raw_labels = {v: raw.nodes[v]['value'] for v in raw.nodes()}
    G, orig_nodes = _prepare(raw)
    true_labels = [raw_labels[v] for v in orig_nodes]
    return G, true_labels


def load_gml(path):
    """Carrega GML genérico sem ground truth. Retorna apenas G."""
    G, _ = _prepare(nx.read_gml(path))
    return G


def generate_ad_hoc_network(n=128, num_communities=4, z_out=3, seed=42):
    """
    Rede ad hoc do artigo: 128 nós, 4 comunidades, grau médio fixo d=16.
    z_out varia de 0.5 a 8 no experimento da Fig. 1.
    Retorna (G, true_labels).
    """
    d = 16
    nodes_per_comm = n // num_communities
    p_out = z_out / (n - nodes_per_comm)
    p_in = (d - z_out) / (nodes_per_comm - 1)

    sizes = [nodes_per_comm] * num_communities
    probs = [[p_in if i == j else p_out for j in range(num_communities)]
             for i in range(num_communities)]

    raw = nx.stochastic_block_model(sizes, probs, seed=seed)
    raw_labels = {v: v // nodes_per_comm for v in raw.nodes()}
    G, orig_nodes = _prepare(raw)
    true_labels = [raw_labels[v] for v in orig_nodes]
    return G, true_labels


def generate_gaussian_network(n=400, dist_threshold=0.8, seed=42):
    """
    Rede sintética com 3 comunidades por mistura gaussiana (400 nós).
    Retorna (G, points, true_labels).
    """
    rng = np.random.default_rng(seed)
    means = [np.array([1.0, 4.0]), np.array([2.5, 5.5]), np.array([0.5, 6.0])]
    sizes = [100, 150, 150]
    sigma = 0.15 * np.eye(2)

    parts = [rng.multivariate_normal(mu, sigma, s) for mu, s in zip(means, sizes)]
    points = np.vstack(parts)

    raw = nx.Graph()
    raw.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) <= dist_threshold:
                raw.add_edge(i, j)

    raw_labels = {v: k for k, _ in enumerate(sizes) for v in range(sum(sizes[:k]), sum(sizes[:k+1]))}
    G, orig_nodes = _prepare(raw)
    true_labels = [raw_labels[v] for v in orig_nodes]
    points_reindexed = points[orig_nodes]
    return G, points_reindexed, true_labels


def evaluate_ari(true_labels, pred_labels):
    """
    Adjusted Rand Index entre ground truth e labels preditos.
    ARI=1 perfeito, ARI=0 aleatório, ARI<0 pior que aleatório.
    """
    return adjusted_rand_score(true_labels, pred_labels)


def _prepare(G):
    """
    Extrai o maior componente conexo e renomeia nós como inteiros 0..n-1.
    Retorna (G_prepared, orig_nodes) onde orig_nodes[i] é o nó original
    que virou o nó i — necessário para alinhar true_labels após renomeação.
    """
    lcc_nodes = max(nx.connected_components(G), key=len)
    subG = G.subgraph(lcc_nodes).copy()
    orig_nodes = list(subG.nodes())
    mapping = {v: i for i, v in enumerate(orig_nodes)}
    return nx.relabel_nodes(subG, mapping), orig_nodes
