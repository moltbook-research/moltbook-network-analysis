import pandas as pd
import networkx as nx
from tqdm import tqdm
from itertools import combinations

from config import Config

def build_answer_graph(df: pd.DataFrame) -> nx.Graph:
    print("\nBuilding Answer graph...")

    G = nx.Graph()
    comment_to_agent = dict(zip(df["comment_id"], df["agent_name_y"]))

    for parent, child in tqdm(zip(df["parent_id"], df["agent_name_y"]), total=len(df)):
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


def build_discussion_graph(df: pd.DataFrame, cfg: Config) -> nx.Graph:
    print("\nBuilding Discussion graph...")

    G = nx.Graph()
    df = df.copy()
    df["root"] = df["parent_id"].fillna(df["comment_id"])
    grouped = df.groupby("root")["agent_name_y"]

    for _, agents in tqdm(grouped):
        unique_agents = list(set(agents))

        if len(unique_agents) > cfg.max_thread_size:
            continue

        for u, v in combinations(unique_agents, 2):
            G.add_edge(u, v)

    print("Discussion Nodes:", G.number_of_nodes())
    print("Discussion Edges:", G.number_of_edges())
    return G


def clean_graph(G: nx.Graph, cfg: Config) -> nx.Graph:
    low_nodes = [n for n, d in dict(G.degree()).items() if d < cfg.min_degree_filter]
    G.remove_nodes_from(low_nodes)

    if G.number_of_nodes() > 0:
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    print("After cleaning - Nodes:", G.number_of_nodes())
    print("After cleaning - Edges:", G.number_of_edges())
    return G