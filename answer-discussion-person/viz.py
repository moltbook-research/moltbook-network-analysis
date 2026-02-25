import os
import networkx as nx
import matplotlib.pyplot as plt

from config import Config

def draw_graph(G: nx.Graph, filename: str, title: str, cfg: Config) -> None:
    print("\nComputing layout...")

    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    plt.figure(figsize=(10, 10))

    degrees = dict(G.degree())
    node_sizes = [degrees[n] * 20 for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, edge_color="black", alpha=0.6, width=0.8)
    nx.draw_networkx_nodes(G, pos, node_color="red", node_size=node_sizes, alpha=0.9)

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    path = os.path.join(cfg.results_dir, filename)
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)