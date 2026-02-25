import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset

from fa2_modified import ForceAtlas2
import community as community_louvain


# =====================================================
# CONFIG
# =====================================================

RESULTS_DIR = "moltbook_forceatlas2"
os.makedirs(RESULTS_DIR, exist_ok=True)

ITERATIONS = 1500
MIN_EDGE_WEIGHT = 2   # remove weak interactions
KEEP_ONLY_GIANT_COMPONENT = True


# =====================================================
# LOAD DATA
# =====================================================

def load_moltbook():
    print("Loading Moltbook dataset...")

    comments = load_dataset(
        "SimulaMet/moltbook-observatory-archive",
        "comments",
        split="archive"
    ).to_pandas()

    comments = comments.rename(columns={"id": "comment_id"})

    agents = load_dataset(
        "SimulaMet/moltbook-observatory-archive",
        "agents",
        split="archive"
    ).to_pandas()

    agents = agents[["id", "name"]]
    agents = agents.rename(columns={
        "id": "agent_id",
        "name": "agent_name_y"
    })

    df = comments.merge(agents, on="agent_id", how="left")

    print("Rows:", len(df))
    print("Unique agents:", df["agent_name_y"].nunique())

    return df


# =====================================================
# BUILD WEIGHTED REPLY NETWORK
# =====================================================

def build_reply_network(df):

    print("\nBuilding weighted reply network...")

    G = nx.Graph()

    comment_to_agent = dict(zip(df["comment_id"], df["agent_name_y"]))

    for _, row in tqdm(df.iterrows(), total=len(df)):

        parent = row["parent_id"]
        child_agent = row["agent_name_y"]

        if pd.isna(parent) or pd.isna(child_agent):
            continue

        if parent not in comment_to_agent:
            continue

        parent_agent = comment_to_agent[parent]

        if pd.isna(parent_agent):
            continue

        if child_agent == parent_agent:
            continue

        a = str(child_agent)
        b = str(parent_agent)

        if G.has_edge(a, b):
            G[a][b]["weight"] += 1
        else:
            G.add_edge(a, b, weight=1)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    return G


# =====================================================
# CLEAN GRAPH
# =====================================================

def clean_graph(G):

    print("\nCleaning graph...")

    # Remove weak edges
    edges_to_remove = [
        (u, v) for u, v, d in G.edges(data=True)
        if d["weight"] < MIN_EDGE_WEIGHT
    ]
    G.remove_edges_from(edges_to_remove)

    # Remove isolates
    G.remove_nodes_from(list(nx.isolates(G)))

    # Keep only giant component
    if KEEP_ONLY_GIANT_COMPONENT:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    print("Nodes after cleaning:", G.number_of_nodes())
    print("Edges after cleaning:", G.number_of_edges())

    return G


# =====================================================
# COMMUNITY DETECTION
# =====================================================

def detect_communities(G):

    print("\nDetecting communities (Louvain)...")

    partition = community_louvain.best_partition(G, weight="weight")

    print("Communities found:", len(set(partition.values())))

    return partition


# =====================================================
# FORCEATLAS2 LAYOUT (CLUSTER-OPTIMIZED)
# =====================================================

def compute_forceatlas2(G):

    print("\nComputing ForceAtlas2 layout...")

    forceatlas2 = ForceAtlas2(
        outboundAttractionDistribution=True,
        linLogMode=False,               # MUST be False in fa2_modified
        adjustSizes=False,
        edgeWeightInfluence=2.0,
        jitterTolerance=0.5,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        scalingRatio=8.0,               # strong separation
        strongGravityMode=False,
        gravity=1.5,
        verbose=True
    )

    pos = forceatlas2.forceatlas2_networkx_layout(
        G,
        iterations=ITERATIONS
    )

    return pos


# =====================================================
# DRAW GRAPH
# =====================================================

def draw_graph(G, pos, partition):

    print("\nDrawing graph...")

    plt.figure(figsize=(16, 16))

    cmap = plt.cm.get_cmap("tab20")

    node_colors = [
        cmap(partition[node] % 20)
        for node in G.nodes()
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.05,
        width=0.3
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=20,
        linewidths=0
    )

    plt.title("Moltbook Reply Network\nForceAtlas2 + Louvain (Clustered)",
              fontsize=15)

    plt.axis("off")
    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "moltbook_forceatlas2_CLUSTERED.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved to:", output_path)


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n======================================")
    print("MOLTBOOK FORCEATLAS2 COMMUNITY GRAPH")
    print("CLUSTERED VERSION")
    print("======================================\n")

    df = load_moltbook()
    G = build_reply_network(df)
    G = clean_graph(G)

    if G.number_of_nodes() == 0:
        print("Graph empty after cleaning.")
        return

    partition = detect_communities(G)
    pos = compute_forceatlas2(G)
    draw_graph(G, pos, partition)

    print("\nDone.\n")


if __name__ == "__main__":
    main()