import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
from datasets import load_dataset


# ======================================================
# CONFIG
# ======================================================

RESULTS_DIR = "answer_discussion_person"
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_THREAD_SIZE = 40
MIN_DEGREE_FILTER = 2


# ======================================================
# LOAD FULL DATA
# ======================================================

def load_moltbook():

    print("\nLoading FULL Moltbook dataset...\n")

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
    df = df.dropna(subset=["agent_name_y"])

    print("Rows:", len(df))
    print("Unique agents:", df["agent_name_y"].nunique())

    return df


# ======================================================
# ANSWER GRAPH (DIRECT REPLIES)
# ======================================================

def build_answer_graph(df):

    print("\nBuilding Answer graph...")

    G = nx.Graph()

    comment_to_agent = dict(zip(df["comment_id"], df["agent_name_y"]))

    for parent, child in tqdm(
        zip(df["parent_id"], df["agent_name_y"]),
        total=len(df)
    ):

        if pd.isna(parent):
            continue

        if parent not in comment_to_agent:
            continue

        parent_agent = comment_to_agent[parent]

        if parent_agent != child:
            G.add_edge(parent_agent, child)

    print("Answer Nodes:", G.number_of_nodes())
    print("Answer Edges:", G.number_of_edges())

    return G


# ======================================================
# DISCUSSION GRAPH
# ======================================================

def build_discussion_graph(df):

    print("\nBuilding Discussion graph...")

    G = nx.Graph()

    df["root"] = df["parent_id"].fillna(df["comment_id"])
    grouped = df.groupby("root")["agent_name_y"]

    for _, agents in tqdm(grouped):

        unique_agents = list(set(agents))

        if len(unique_agents) > MAX_THREAD_SIZE:
            continue

        for u, v in combinations(unique_agents, 2):
            G.add_edge(u, v)

    print("Discussion Nodes:", G.number_of_nodes())
    print("Discussion Edges:", G.number_of_edges())

    return G


# ======================================================
# CLEAN GRAPH
# ======================================================

def clean_graph(G):

    low_nodes = [n for n, d in dict(G.degree()).items() if d < MIN_DEGREE_FILTER]
    G.remove_nodes_from(low_nodes)

    if G.number_of_nodes() > 0:
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    print("After cleaning - Nodes:", G.number_of_nodes())
    print("After cleaning - Edges:", G.number_of_edges())

    return G


# ======================================================
# DRAW GRAPH (NO COLORS, NO COMMUNITIES)
# ======================================================

def draw_graph(G, filename, title):

    print("\nComputing layout...")

    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    plt.figure(figsize=(10, 10))

    degrees = dict(G.degree())
    node_sizes = [degrees[n] * 20 for n in G.nodes()]  # bigger scaling

    # Draw edges (black, visible)
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="black",
        alpha=0.6,
        width=0.8
    )

    # Draw nodes (ALL RED)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="red",
        node_size=node_sizes,
        alpha=0.9
    )

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)


# ======================================================
# MAIN
# ======================================================

def main():

    print("\nGenerating Moltbook Answer & Discussion Graphs...\n")

    df = load_moltbook()

    # -------- ANSWER GRAPH --------
    G_answer = build_answer_graph(df)
    G_answer = clean_graph(G_answer)
    draw_graph(
        G_answer,
        "answer_person_graph.png",
        "Moltbook Answer Person Graph"
    )

    # -------- DISCUSSION GRAPH --------
    G_disc = build_discussion_graph(df)
    G_disc = clean_graph(G_disc)
    draw_graph(
        G_disc,
        "discussion_person_graph.png",
        "Moltbook Discussion Person Graph"
    )

    print("\nDone.\n")


if __name__ == "__main__":
    main()