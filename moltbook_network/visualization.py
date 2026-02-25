import os
import matplotlib.pyplot as plt
import networkx as nx


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# COMMUNITY GRAPH
def draw_communities(G, node_to_community, results_dir):

    ensure_dir(results_dir)

    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, k=0.1, seed=42)

    communities = sorted(set(node_to_community.values()))

    for c in communities:
        nodes = [n for n in G.nodes() if node_to_community[n] == c]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_size=15,
            node_color=[plt.cm.tab20(c % 20)],
            label=f"Community {c}"
        )

    nx.draw_networkx_edges(G, pos, alpha=0.1)

    plt.legend(fontsize=7)
    plt.title("Network Colored by Community", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "network_communities.png"), dpi=300)
    plt.close()


#   STRUCTURAL ROLES
def draw_roles(G, role_labels, results_dir):

    ensure_dir(results_dir)

    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, k=0.1, seed=42)

    role_colors = {
        "Hub": "red",
        "Bridge": "orange",
        "Peripheral": "lightblue"
    }

    for role, color in role_colors.items():
        nodes = [n for n in G.nodes() if role_labels[n] == role]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=color,
            node_size=20,
            label=role
        )

    nx.draw_networkx_edges(G, pos, alpha=0.1)

    plt.legend()
    plt.title("Network Colored by Structural Role", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "network_roles.png"), dpi=300)
    plt.close()


# DEGREE GRAPH WITH TOP HUBS LABELED

def draw_degree_with_labels(G, results_dir, top_k=5):

    ensure_dir(results_dir)

    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, k=0.1, seed=42)

    degrees = dict(G.degree())
    node_sizes = [degrees[n] * 5 for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.1)

    # Top hubs
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_k]

    labels = {n: n for n in top_nodes}

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8,
        font_color="darkred"
    )

    plt.title(f"Node Size = Degree (Top {top_k} Hubs Labeled)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "network_degree.png"), dpi=300)
    plt.close()


# TOP HUB EGO NETWORK

def draw_top_hub_ego(G, results_dir):

    ensure_dir(results_dir)

    degrees = dict(G.degree())
    top_node = max(degrees, key=degrees.get)

    ego = nx.ego_graph(G, top_node)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(ego, seed=42)

    nx.draw(
        ego,
        pos,
        node_size=200,
        with_labels=False,
        font_size=8
    )

    plt.title(
        f"Ego Network of Top Hub: {top_node}\nDegree = {degrees[top_node]}",
        fontsize=13
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ego_top_hub.png"), dpi=300)
    plt.close()

def label_structural_roles(G, hub_threshold=50, bridge_threshold=0.01):
    """
    Assign structural roles to nodes:
    - Hub: high degree
    - Bridge: high betweenness
    - Peripheral: everything else
    """

    roles = {}

    degree_dict = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)

    for node in G.nodes():

        if degree_dict[node] >= hub_threshold:
            roles[node] = "Hub"

        elif betweenness[node] >= bridge_threshold:
            roles[node] = "Bridge"

        else:
            roles[node] = "Peripheral"

    hub_count = sum(1 for r in roles.values() if r == "Hub")
    bridge_count = sum(1 for r in roles.values() if r == "Bridge")
    peripheral_count = sum(1 for r in roles.values() if r == "Peripheral")

    print("\nSTRUCTURAL ROLE SUMMARY")
    print("Hubs:", hub_count)
    print("Bridges:", bridge_count)
    print("Peripheral:", peripheral_count)

    return roles