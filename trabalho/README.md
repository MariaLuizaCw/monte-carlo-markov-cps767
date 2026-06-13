# Detecção de Comunidades em Redes Complexas com SADI

## Introdução

Este trabalho foi inspirado no artigo:

> Jian Liu, Tingzhan Liu,
> **Detecting community structure in complex networks using simulated annealing with k-means algorithms**,
> *Physica A: Statistical Mechanics and its Applications*,
> Volume 389, Issue 11, 2010, Pages 2300–2309, ISSN 0378-4371.
> https://doi.org/10.1016/j.physa.2010.01.042

O artigo original propõe o algoritmo **SADI** (*Simulated Annealing with Dissimilarity Index*), que combina *Simulated Annealing* com k-means baseado em índice de dissimilaridade para detectar comunidades em redes complexas, utilizando a **modularidade Q** de Newman-Girvan como função objetivo.

O objetivo principal deste trabalho é **replicar os experimentos do artigo**, substituindo a métrica de otimização original  (modularidade Q) pelo **Silhouette Score**, e avaliar o impacto dessa troca nos resultados. Nos datasets em que existe ground truth disponível, os resultados são comparados quantitativamente por meio do **Adjusted Rand Index (ARI)**, permitindo medir a qualidade da partição descoberta em relação à estrutura real conhecida.

---

## Datasets Utilizados

| Dataset | Tipo | Nós | Arestas | Descrição | Ground Truth | Nº Comunidades (GT) |
|---|---|---|---|---|---|---|
| Karate Club | Real | 34 | 78 | Rede social de um clube de karatê universitário. Após uma disputa interna, o clube se dividiu em dois grupos. | Sim | 2 |
| Dolphins | Real | 62 | 159 | Associações sociais frequentes entre golfinhos em Doubtful Sound, Nova Zelândia. | Parcial | 2 |
| Political Books | Real | 105 | 441 | Rede de co-compra de livros políticos na Amazon. Arestas ligam livros comprados juntos. | Sim | 3 |
| Les Misérables | Real | 77 | 254 | Interações entre personagens do romance de Victor Hugo. Arestas representam co-aparição em cenas. | Não | — |
| Football | Real | 115 | 613 | Rede de jogos disputados entre equipes universitárias de futebol americano da Divisão IA na temporada regular de outono de 2000. | Sim | 12 |
| Ad Hoc | Sintético | 128 | var | Benchmark clássico com 4 comunidades de 32 nós. Grau médio fixo em 16. Dificuldade controlada pelo parâmetro z_out. | Sim | 4 |
| Gaussian Mixture | Sintético | 400 | var | Rede gerada a partir de mistura de 3 gaussianas em 2D. Arestas criadas por limiar de distância euclidiana. | Sim | 3 |

---

## Estrutura do Projeto

```
trabalho/
│
├── experimento.ipynb          # Notebook principal com todos os experimentos
├── animate_sadi.py            # Script para gerar animação GIF do algoritmo SADI
│
├── modules/
│   ├── __init__.py
│   ├── markov.py              # Cadeia de Markov e matriz de dissimilaridade
│   ├── sadi.py                # Implementação do algoritmo SADI
│   ├── data_loader.py         # Carregamento dos datasets e avaliação com ARI
│   └── visualization.py       # Funções de visualização e geração de gráficos
│
├── data/                      # Datasets no formato GML comprimidos em zip
│   ├── karate.zip
│   ├── dolphins.zip
│   ├── polbooks.zip
│   ├── lesmis.zip
│   └── football.zip
│
└── images/                    # Imagens e GIFs gerados pelos experimentos
    ├── sadi_animation.gif
    ├── cooling_schedule_comparison.png
    ├── n_communities_comparison.png
    └── ari_comparison.png
```

### Descrição dos arquivos

1. **`experimento.ipynb`** — Notebook principal que orquestra todos os experimentos: aplica o SADI com Silhouette Score e com modularidade Q em cada dataset, compara os três esquemas de resfriamento e reporta ARI nos datasets com ground truth.

2. **`animate_sadi.py`** — Script standalone que executa uma versão instrumentada do SADI em um grafo sintético pequeno e gera um GIF animado mostrando as 5 fases do algoritmo a cada iteração: Inicialização, Geração de Vizinhos, Refinamento, Avaliação/Aceitação e Resfriamento.

   ```bash
   python animate_sadi.py                        # salva images/sadi_animation.gif
   python animate_sadi.py --show                 # abre janela interativa
   python animate_sadi.py --fps 2 --max-frames 60
   ```

3. **`modules/markov.py`** — Constrói a cadeia de Markov a partir do grafo, calcula a matriz de *mean first-passage times* (MFPT) resolvendo o sistema linear para cada nó destino, e deriva a matriz de dissimilaridade Δ(x, y).

4. **`modules/sadi.py`** — Implementação completa do algoritmo SADI. Inclui atribuição de comunidades por k-means com índice de dissimilaridade, cálculo dos centros e da força de comunidade, operações de perturbação (retain/delete/split), critério de aceitação de Metropolis, e suporte às funções objetivo **Silhouette Score** e **modularidade Q**. Também implementa três esquemas de resfriamento: geométrico, linear e logarítmico.

5. **`modules/data_loader.py`** — Carrega cada dataset (de arquivo zip ou GML), extrai o maior componente conexo e padroniza os índices dos nós. Também gera as redes sintéticas *ad hoc* e *Gaussian Mixture*. Fornece a função `evaluate_ari` para comparação com ground truth via Adjusted Rand Index.

6. **`modules/visualization.py`** — Funções de visualização: grafo colorido por comunidades, comparação lado a lado com ground truth e ARI, heatmap da matriz de dissimilaridade, evolução do score ao longo das iterações, comparação entre funções de resfriamento e estatísticas de visita por número de comunidades.

7. **`data/`** — Datasets no formato GML comprimidos em zip: `karate.zip`, `dolphins.zip`, `polbooks.zip`, `lesmis.zip` e `football.zip`.

8. **`images/`** — Imagens e GIFs gerados pelos experimentos: `sadi_animation.gif`, `cooling_schedule_comparison.png`, `n_communities_comparison.png` e `ari_comparison.png`.
