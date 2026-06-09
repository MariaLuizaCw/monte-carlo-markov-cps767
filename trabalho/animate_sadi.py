"""
Animação do algoritmo SADI — Simulated Annealing with Dissimilarity Index.
Gera um GIF mostrando as 5 etapas principais do algoritmo passo a passo.

Uso:
    python animate_sadi.py                      # salva sadi_animation.gif
    python animate_sadi.py --output meu.gif     # nome customizado
    python animate_sadi.py --show               # abre janela interativa
    python animate_sadi.py --fps 3 --max-frames 100
"""

import argparse
import random
import math
import sys
import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(__file__))
from modules.sadi import (
    assign_communities, compute_center,
    community_strength, compute_silhouette, _make_cooling_fn,
)

# ── Paleta ───────────────────────────────────────────────────────────────────

PHASE_COLORS = {
    1: '#4A90D9',
    2: '#E67E22',
    3: '#27AE60',
    4: '#8E44AD',
    5: '#C0392B',
}
PHASE_NAMES = {
    1: 'Fase 1 — Inicialização',
    2: 'Fase 2 — Geração de Vizinhos',
    3: 'Fase 3 — Refinamento',
    4: 'Fase 4 — Avaliação e Aceitação',
    5: 'Fase 5 — Resfriamento',
}
PHASE_SHORT = ['Inicialização', 'Vizinhos', 'Refinamento', 'Avaliação', 'Resfriamento']

COMM_PALETTE = [
    '#E74C3C', '#3498DB', '#2ECC71', '#F39C12',
    '#9B59B6', '#1ABC9C', '#E67E22', '#34495E',
]
OP_LABEL = {
    'retain': 'RETAIN', 'delete': 'DELETE', 'split': 'SPLIT',
    'init': 'INICIALIZAÇÃO', 'cool': 'RESFRIAMENTO',
}
OP_COLOR = {
    'retain': '#3498DB', 'delete': '#E74C3C', 'split': '#F39C12',
    'init': '#27AE60',   'cool':  '#95A5A6',
}


# ── Grafo e dissimilaridade ──────────────────────────────────────────────────

def make_test_graph(n_per_comm=5, n_communities=4, p_in=0.72, p_out=0.05, seed=42):
    rng = random.Random(seed)
    n = n_per_comm * n_communities
    G = nx.Graph()
    G.add_nodes_from(range(n))
    true_labels = [c for c in range(n_communities) for _ in range(n_per_comm)]
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if true_labels[i] == true_labels[j] else p_out
            if rng.random() < p:
                G.add_edge(i, j)
    return G, np.array(true_labels)


def compute_dissimilarity(G):
    n = len(G.nodes())
    A = nx.to_numpy_array(G)
    D = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i, j] = 0.0
            else:
                common = float(np.dot(A[i], A[j]))
                denom  = A[i].sum() + A[j].sum() - common
                D[i, j] = 1.0 - (common / denom) if denom > 0 else 1.0
    return D


# ── Coleta de frames ─────────────────────────────────────────────────────────

def collect_frames(G, Delta,
                   T_max=2.0, T_min=0.25,
                   alpha=0.5, R=2,
                   N_min=2, N_max=None,
                   max_frames=100,
                   seed=42):
    """
    Executa o SADI instrumentado e devolve uma lista de frames (dicts).

    Cada frame contém:
        phase, centers, labels, op, T, score, best_score,
        n_comm, proposed_centers, accepted, highlight_comm
    """
    random.seed(seed)
    np.random.seed(seed)

    n = len(G.nodes())
    if N_max is None:
        N_max = max(2, n // 3)

    score_fn = lambda lbl: compute_silhouette(Delta, lbl)
    nodes = list(range(n))
    frames = []

    def snap(phase, centers, labels, **kw):
        frames.append({
            'phase':            phase,
            'centers':          list(centers),
            'labels':           np.array(labels),
            'T':                kw.get('T', T_max),
            'score':            kw.get('score', 0.0),
            'best_score':       kw.get('best_score', 0.0),
            'n_comm':           kw.get('n_comm', len(centers)),
            'op':               kw.get('op', 'retain'),
            'accepted':         kw.get('accepted', True),
            'highlight_comm':   kw.get('highlight_comm', None),
            'proposed_centers': kw.get('proposed_centers', list(centers)),
        })

    # Fase 1 — Inicialização
    k0 = random.randint(N_min, min(N_max, len(nodes)))
    centers = random.sample(nodes, k0)
    communities, labels = assign_communities(Delta, centers)
    energy = -score_fn(labels)
    best_energy = energy
    best_labels = labels.copy()
    best_centers = centers.copy()

    snap(1, centers, labels,
         T=T_max, score=-energy, best_score=-best_energy,
         n_comm=len(centers), op='init')

    schedule = _make_cooling_fn('geometric', T_max, T_min, alpha)
    temp_step = 0
    T = schedule(temp_step)

    while T > T_min and len(frames) < max_frames:
        for _ in range(R):
            if len(frames) >= max_frames:
                break

            op = random.choice(['retain', 'delete', 'split'])
            new_centers = centers.copy()
            highlight = None

            if op == 'delete' and len(new_centers) > N_min:
                strengths = {k: community_strength(G, communities[k]) for k in communities}
                weakest = min(strengths, key=strengths.get)
                highlight = weakest
                if weakest < len(new_centers):
                    new_centers.pop(weakest)

            elif op == 'split' and len(new_centers) < N_max:
                avg_s = {
                    k: community_strength(G, communities[k]) / len(communities[k])
                    for k in communities if len(communities[k]) > 1
                }
                if avg_s:
                    split_k = min(avg_s, key=avg_s.get)
                    highlight = split_k
                    split_center = new_centers[split_k]
                    others = [x for x in communities[split_k] if x != split_center]
                    if others:
                        new_c = max(others, key=lambda x: Delta[x, split_center])
                        new_centers.append(new_c)

            # Fase 2 — proposta gerada
            snap(2, centers, labels,
                 T=T, score=-energy, best_score=-best_energy,
                 n_comm=len(centers), op=op,
                 highlight_comm=highlight,
                 proposed_centers=new_centers)

            # Fase 3 — refinamento
            new_comm, new_labels = assign_communities(Delta, new_centers)
            new_centers_refined = [
                compute_center(new_comm[k], Delta) for k in range(len(new_centers))
            ]
            new_comm, new_labels = assign_communities(Delta, new_centers_refined)

            snap(3, new_centers_refined, new_labels,
                 T=T, score=score_fn(new_labels), best_score=-best_energy,
                 n_comm=len(new_centers_refined), op=op,
                 highlight_comm=highlight)

            new_energy = -score_fn(new_labels)
            delta_E = new_energy - energy
            prob = math.exp(-delta_E / T) if delta_E >= 0 and T > 0 else 1.0
            accepted = delta_E < 0 or random.random() < prob

            # Fase 4 — avaliação
            snap(4,
                 new_centers_refined if accepted else centers,
                 new_labels          if accepted else labels,
                 T=T, score=-new_energy, best_score=-best_energy,
                 n_comm=len(new_centers_refined if accepted else centers),
                 op=op, accepted=accepted)

            if accepted:
                centers, communities, labels, energy = (
                    new_centers_refined, new_comm, new_labels, new_energy)
                if energy < best_energy:
                    best_energy  = energy
                    best_labels  = labels.copy()
                    best_centers = centers.copy()

        # Fase 5 — resfriamento
        temp_step += 1
        T = schedule(temp_step)
        snap(5, centers, labels,
             T=T, score=-energy, best_score=-best_energy,
             n_comm=len(centers), op='cool')

    return frames


# ── Renderização de um frame ─────────────────────────────────────────────────

def draw_frame(fig, ax_graph, ax_info, ax_phases, frame, pos, G, T_max, T_min):
    for ax in (ax_graph, ax_info, ax_phases):
        ax.clear()

    phase    = frame['phase']
    labels   = frame['labels']
    centers  = frame['centers']
    proposed = frame['proposed_centers']
    op       = frame['op']
    T        = frame['T']
    score    = frame['score']
    best_sc  = frame['best_score']
    n_comm   = frame['n_comm']
    accepted = frame['accepted']
    highlight = frame['highlight_comm']
    ph_color  = PHASE_COLORS[phase]

    # ── Grafo ──────────────────────────────────────────────────────────────
    n_comm_cur = max(len(set(labels)), 1)
    node_colors = [COMM_PALETTE[int(labels[nd]) % len(COMM_PALETTE)] for nd in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax_graph,
                           alpha=0.22, edge_color='#666666', width=0.9)

    non_centers = [nd for nd in G.nodes() if nd not in centers]
    if non_centers:
        nx.draw_networkx_nodes(G, pos, nodelist=non_centers,
                               node_color=[node_colors[nd] for nd in non_centers],
                               node_size=300, ax=ax_graph, alpha=0.88)

    if centers:
        nx.draw_networkx_nodes(G, pos, nodelist=centers,
                               node_color=[node_colors[nd] for nd in centers],
                               node_size=540, ax=ax_graph,
                               edgecolors='black', linewidths=2.2)

    # Fase 2: centros propostos novos em amarelo
    if phase == 2:
        new_only = [c for c in proposed if c not in centers]
        if new_only:
            nx.draw_networkx_nodes(G, pos, nodelist=new_only,
                                   node_color='#FFEB3B', node_size=540, ax=ax_graph,
                                   edgecolors='#F57F17', linewidths=2.5, alpha=0.9)

    # Comunidade destacada (alvo da operação)
    if highlight is not None and phase in (2, 3):
        affected = [nd for nd in G.nodes() if int(labels[nd]) == highlight]
        if affected:
            nx.draw_networkx_nodes(G, pos, nodelist=affected,
                                   node_color='none', node_size=460, ax=ax_graph,
                                   edgecolors=ph_color, linewidths=3.2, alpha=0.75)

    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax_graph,
                            font_color='white', font_weight='bold')

    # Legenda comunidades
    patches = [
        mpatches.Patch(color=COMM_PALETTE[k % len(COMM_PALETTE)], label=f'C{k+1}')
        for k in range(n_comm_cur)
    ]
    ax_graph.legend(handles=patches, loc='lower left', fontsize=7.5,
                    framealpha=0.85, ncol=min(4, n_comm_cur))

    title = PHASE_NAMES[phase]
    if op not in ('init', 'cool', 'retain'):
        title += f'  [{OP_LABEL.get(op, op)}]'
    ax_graph.set_title(title, fontsize=11, color=ph_color, fontweight='bold', pad=9)
    ax_graph.axis('off')

    # ── Painel lateral ─────────────────────────────────────────────────────
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis('off')

    # Barra de temperatura
    t_ratio = max(0.0, min(1.0, (T - T_min) / max(T_max - T_min, 1e-9)))
    ax_info.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.84), 0.90, 0.09,
        boxstyle='round,pad=0.01', facecolor='#EEEEEE', edgecolor='#CCCCCC'))
    if t_ratio > 0.01:
        ax_info.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.84), 0.90 * t_ratio, 0.09,
            boxstyle='round,pad=0.01', facecolor='#E74C3C', edgecolor='none', alpha=0.85))
    ax_info.text(0.50, 0.885, f'T = {T:.4f}',
                 ha='center', va='center', fontsize=9, fontweight='bold',
                 color='white' if t_ratio > 0.25 else '#444444')
    ax_info.text(0.50, 0.96, 'Temperatura',
                 ha='center', va='center', fontsize=8, color='#666666')

    # Score atual
    ax_info.text(0.50, 0.755, 'Score atual', ha='center', fontsize=8, color='#888888')
    sc_color = '#27AE60' if score >= 0 else '#E74C3C'
    ax_info.text(0.50, 0.675, f'{score:.4f}',
                 ha='center', fontsize=14, fontweight='bold', color=sc_color)

    # Melhor score
    ax_info.text(0.50, 0.595, 'Melhor score', ha='center', fontsize=8, color='#888888')
    ax_info.text(0.50, 0.515, f'{best_sc:.4f}',
                 ha='center', fontsize=14, fontweight='bold', color='#2C3E50')

    # Número de comunidades
    ax_info.text(0.50, 0.435, 'Comunidades', ha='center', fontsize=8, color='#888888')
    ax_info.text(0.50, 0.355, str(n_comm),
                 ha='center', fontsize=20, fontweight='bold', color=ph_color)

    # Aceito / Rejeitado (fase 4)
    if phase == 4:
        txt = '✓  ACEITA' if accepted else '✗  REJEITADA'
        clr = '#27AE60' if accepted else '#E74C3C'
        ax_info.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.245), 0.90, 0.075,
            boxstyle='round,pad=0.02', facecolor=clr, edgecolor='none', alpha=0.15))
        ax_info.text(0.50, 0.283, txt,
                     ha='center', va='center', fontsize=9.5,
                     fontweight='bold', color=clr)

    # Operação
    op_clr = OP_COLOR.get(op, '#555555')
    ax_info.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.07), 0.90, 0.12,
        boxstyle='round,pad=0.02', facecolor=op_clr, edgecolor='none', alpha=0.15))
    ax_info.text(0.50, 0.135, OP_LABEL.get(op, op.upper()),
                 ha='center', va='center', fontsize=9, fontweight='bold', color=op_clr)
    ax_info.text(0.50, 0.04, 'operação',
                 ha='center', va='center', fontsize=7, color='#AAAAAA')

    # ── Barra de fases (rodapé) ─────────────────────────────────────────────
    ax_phases.set_xlim(0, 5)
    ax_phases.set_ylim(0, 1)
    ax_phases.axis('off')
    for i, (name, clr) in enumerate(zip(PHASE_SHORT, PHASE_COLORS.values())):
        active = (i + 1 == phase)
        bg  = clr     if active else '#DDDDDD'
        txc = 'white' if active else '#999999'
        ax_phases.add_patch(mpatches.FancyBboxPatch(
            (i + 0.04, 0.12), 0.92, 0.76,
            boxstyle='round,pad=0.05',
            facecolor=bg, edgecolor='none', alpha=0.9 if active else 0.55))
        ax_phases.text(i + 0.50, 0.62, str(i + 1),
                       ha='center', va='center', fontsize=9,
                       fontweight='bold', color=txc)
        ax_phases.text(i + 0.50, 0.27, name,
                       ha='center', va='center', fontsize=6.2, color=txc)


# ── Construção da animação ────────────────────────────────────────────────────

def build_animation(frames, G, pos, T_max, T_min, interval_ms=600):
    fig = plt.figure(figsize=(13, 7.5), facecolor='#F8F9FA')
    gs  = gridspec.GridSpec(2, 2,
                            height_ratios=[9, 1],
                            width_ratios=[7, 3],
                            hspace=0.10, wspace=0.14)
    ax_graph  = fig.add_subplot(gs[0, 0])
    ax_info   = fig.add_subplot(gs[0, 1])
    ax_phases = fig.add_subplot(gs[1, :])

    for ax in (ax_graph, ax_info, ax_phases):
        ax.set_facecolor('#F8F9FA')

    def update(i):
        draw_frame(fig, ax_graph, ax_info, ax_phases,
                   frames[i], pos, G, T_max, T_min)

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=interval_ms, blit=False)
    return fig, anim


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Animação do algoritmo SADI')
    parser.add_argument('--output', default='images/sadi_animation.gif',
                        help='Arquivo de saída (padrão: images/sadi_animation.gif)')
    parser.add_argument('--show', action='store_true',
                        help='Exibir janela interativa em vez de salvar')
    parser.add_argument('--fps', type=float, default=1,
                        help='Frames por segundo (padrão: 1)')
    parser.add_argument('--interval-ms', type=int, default=None,
                        help='Tempo por frame em ms (substitui --fps; padrão: None)')
    parser.add_argument('--max-frames', type=int, default=80,
                        help='Máximo de frames a capturar (padrão: 80)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semente aleatória (padrão: 42)')
    parser.add_argument('--n-per-comm', type=int, default=5,
                        help='Nós por comunidade no grafo de teste (padrão: 5)')
    parser.add_argument('--n-communities', type=int, default=4,
                        help='Número de comunidades no grafo de teste (padrão: 4)')
    args = parser.parse_args()

    T_MAX, T_MIN = 0.5, 0.01

    print('Gerando grafo de teste...')
    G, _ = make_test_graph(n_per_comm=args.n_per_comm,
                           n_communities=args.n_communities,
                           seed=args.seed)
    print(f'  {len(G.nodes())} nós, {len(G.edges())} arestas.')

    print('Calculando matriz de dissimilaridade...')
    Delta = compute_dissimilarity(G)

    pos = nx.spring_layout(G, seed=args.seed)

    print('Executando SADI e coletando frames...')
    frames = collect_frames(G, Delta,
                            T_max=T_MAX, T_min=T_MIN,
                            alpha=0.5, R=2,
                            max_frames=args.max_frames,
                            seed=args.seed)
    print(f'  {len(frames)} frames coletados.')

    print('Construindo animação...')
    if args.interval_ms is not None:
        interval_ms = args.interval_ms
        fps_out = 1000 / interval_ms
    else:
        fps_out = args.fps
        interval_ms = int(1000 / fps_out)

    fig, anim = build_animation(frames, G, pos,
                                T_max=T_MAX, T_min=T_MIN,
                                interval_ms=interval_ms)

    if args.show:
        plt.show()
    else:
        print(f'Salvando em "{args.output}" ({fps_out:.2f} fps, {interval_ms}ms/frame)...')
        anim.save(args.output, writer=PillowWriter(fps=fps_out), dpi=110)
        print('Concluído.')

    plt.close(fig)


if __name__ == '__main__':
    main()
