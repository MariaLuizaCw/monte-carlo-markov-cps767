import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from scipy.optimize import linear_sum_assignment


def plot_communities(G, labels, centers, title="Detecção de Comunidades — SADI", save_path=None):
    pos = nx.spring_layout(G, seed=42)
    n_communities = len(set(labels))
    colors = cm.Set1(np.linspace(0, 1, n_communities))

    plt.figure(figsize=(10, 7))
    for k in range(n_communities):
        node_list = [v for v in G.nodes() if labels[v] == k]
        nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                               node_color=[colors[k]], node_size=200,
                               label=f'Comunidade {k + 1}')

    nx.draw_networkx_nodes(G, pos, nodelist=centers,
                           node_color='white', node_size=400,
                           edgecolors='black', linewidths=2)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=7)
    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_adhoc_experiment(z_out_values, fractions_correct, save_path=None):
    """Reproduz a Fig. 1 do artigo para a rede ad hoc."""
    plt.figure(figsize=(8, 5))
    plt.plot(z_out_values, fractions_correct, 'o-', label='SADI (Silhouette)')
    plt.xlabel('Out links $z_{out}$')
    plt.ylabel('Fração de nós classificados corretamente')
    plt.title('Ad Hoc Network — Reprodução da Fig. 1 (com Silhouette Score)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_dissimilarity_heatmap(D, labels=None, title="Matriz de Dissimilaridade", save_path=None):
    """Exibe um heatmap de uma matriz de dissimilaridade D (n x n)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(D, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Dissimilaridade')
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_comparison(G, true_labels, pred_labels, centers,
                    title_true="Ground Truth", title_pred="SADI", ari=None, save_path=None):
    """
    Plot lado a lado: comunidades reais vs comunidades detectadas pelo SADI.
    Mostra o ARI no título se fornecido.
    """
    pos = nx.spring_layout(G, seed=42)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, labels, title, show_centers in [
        (axes[0], true_labels, title_true, False),
        (axes[1], pred_labels, title_pred + (f"\nARI = {ari:.4f}" if ari is not None else ""), True),
    ]:
        n_comm = len(set(labels))
        colors = cm.Set1(np.linspace(0, 1, n_comm))
        for k in range(n_comm):
            node_list = [v for v in G.nodes() if labels[v] == k]
            nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                                   node_color=[colors[k]], node_size=200,
                                   label=f'Comunidade {k+1}', ax=ax)
        if show_centers:
            nx.draw_networkx_nodes(G, pos, nodelist=centers,
                                   node_color='white', node_size=400,
                                   edgecolors='black', linewidths=2, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
        ax.set_title(title, fontsize=13)
        ax.legend(loc='upper left', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_silhouette_history(history, title="Evolução do Silhouette Score — SADI", save_path=None):
    """
    Plota a evolução do silhouette score ao longo das iterações do SADI.
    history: lista de tuplas (iteration, temperature, current_silhouette, best_silhouette)
             conforme retornado por sadi_algorithm().
    """
    iterations     = [h[0] for h in history]
    current_scores = [h[2] for h in history]
    best_scores    = [h[3] for h in history]

    _, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(iterations, current_scores, color='steelblue', alpha=0.5, linewidth=0.8, label='Atual')
    ax1.plot(iterations, best_scores,    color='crimson',   linewidth=1.5,            label='Melhor')
    ax1.set_xlabel('Iteração')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title(title)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    temperatures = [h[1] for h in history]
    ax2.plot(iterations, temperatures, color='gray', linewidth=0.8, linestyle='--', alpha=0.6, label='Temperatura')
    ax2.set_ylabel('Temperatura', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_cooling_comparison(histories, labels=None,
                            title="Comparação de Funções de Resfriamento",
                            save_path=None):
    """
    Compara a evolução do melhor silhouette score e da temperatura para
    múltiplas execuções do SADI com diferentes cooling schedules.

    histories : lista de listas de tuplas (iter, T, current_sil, best_sil)
    labels    : nomes de cada schedule (ex.: ['geometric','linear','logarithmic'])
    """
    if labels is None:
        labels = [f"run {i}" for i in range(len(histories))]

    colors = ['steelblue', 'crimson', 'seagreen', 'darkorange']

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

    for hist, lbl, color in zip(histories, labels, colors):
        iters      = [h[0] for h in hist]
        best_scores = [h[3] for h in hist]
        temps       = [h[1] for h in hist]
        ax1.plot(iters, best_scores, color=color, linewidth=1.5, label=lbl)
        ax2.plot(iters, temps,       color=color, linewidth=1.5, label=lbl)

    ax1.set_ylabel('Melhor Silhouette Score')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Iteração')
    ax2.set_ylabel('Temperatura')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_n_communities_stats(histories, labels=None,
                             title="Estatísticas de Visita por Nº de Comunidades",
                             score_label="Melhor Score",
                             save_path=None):
    """
    Para cada número de comunidades N visitado durante o SA, exibe:
      - Contagem de visitas
      - Melhor score (silhouette ou modularity Q)
    comparando os três cooling schedules em gráficos de barras agrupadas.

    histories : lista de listas de tuplas (iter, T, current_score, best_score, n_comm)
    labels    : nomes de cada schedule
    score_label : rótulo do eixo Y para o painel de melhor score
    """
    if labels is None:
        labels = [f"run {i}" for i in range(len(histories))]

    colors = ['steelblue', 'crimson', 'seagreen', 'darkorange']

    stats = {}  # stats[lbl][N] = {'count': int, 'best_score': float}
    all_n = set()
    for hist, lbl in zip(histories, labels):
        stats[lbl] = {}
        for entry in hist:
            _, _, cur_score, _, n = entry
            if n not in stats[lbl]:
                stats[lbl][n] = {'count': 0, 'best_score': -np.inf}
            stats[lbl][n]['count'] += 1
            if cur_score > stats[lbl][n]['best_score']:
                stats[lbl][n]['best_score'] = cur_score
            all_n.add(n)

    n_values = sorted(all_n)
    x = np.arange(len(n_values))
    n_schedules = len(labels)
    width = 0.8 / n_schedules

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(n_values) * 1.5), 5))
    fig.suptitle(title, fontsize=13)

    panels = [
        ('Contagem de Visitas', lambda s: s['count']),
        (score_label,           lambda s: s['best_score'] if s['best_score'] > -np.inf else 0),
    ]

    for ax, (ylabel, getter) in zip(axes, panels):
        for i, (lbl, color) in enumerate(zip(labels, colors)):
            vals = [getter(stats[lbl].get(n, {'count': 0, 'best_score': -np.inf}))
                    for n in n_values]
            offset = (i - n_schedules / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=lbl, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in n_values])
        ax.set_xlabel('Número de Comunidades (N)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def match_labels(pred_labels, true_labels):
    """
    Alinha rótulos preditos com os verdadeiros via Hungarian algorithm.
    Retorna fração de nós corretamente classificados.
    """
    pred = np.array(pred_labels)
    true = np.array(true_labels)
    n_comm = max(pred.max(), true.max()) + 1
    conf = np.zeros((n_comm, n_comm), dtype=int)
    for p, t in zip(pred, true):
        conf[p, t] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    return conf[row_ind, col_ind].sum() / len(true)
