import warnings
warnings.filterwarnings("ignore")

import networkx as nx

from config import (
    SAMPLE_SIZE,
    RESULTS_DIR,
    HUB_DEGREE_THRESHOLD,
    BRIDGE_BETWEENNESS_THRESHOLD
)

from data_loader import load_moltbook
from graph_builder import build_reply_graph, get_largest_component

from visualization import (
    draw_communities,
    draw_roles,
    draw_degree_with_labels,
    draw_top_hub_ego,
    label_structural_roles
)

from analysis import (
    print_basic_stats,
    print_top_degree,
    print_top_in_degree,
    print_top_out_degree,
    analyze_communities
)


def main():

    print("\nMOLTBOOK NETWORK ANALYSIS\n")

    # Load Data
    df = load_moltbook(SAMPLE_SIZE)

    # Build Reply Graph
    G = build_reply_graph(df)

    if G.number_of_nodes() == 0:
        print("Graph is empty. Exiting.")
        return

    # Largest Component
    G_main = get_largest_component(G)

    # ANALYSIS
    print_basic_stats(G_main)
    print_top_degree(G_main, 10)
    print_top_in_degree(G_main, 10)
    print_top_out_degree(G_main, 10)

    # Community detection
    communities = list(nx.algorithms.community.greedy_modularity_communities(G_main))
    analyze_communities(G_main, communities)

    # Build node -> community mapping for visualization
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i

    # Structural roles
    role_labels = label_structural_roles(
        G_main,
        hub_threshold=HUB_DEGREE_THRESHOLD,
        bridge_threshold=BRIDGE_BETWEENNESS_THRESHOLD
    )

    # VISUALIZATIONS
    print("\nGenerating visualizations...\n")

    draw_communities(G_main, node_to_community, RESULTS_DIR)
    draw_roles(G_main, role_labels, RESULTS_DIR)
    draw_degree_with_labels(G_main, RESULTS_DIR, top_k=5)
    draw_top_hub_ego(G_main, RESULTS_DIR)

    print("\nVisualizations saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
