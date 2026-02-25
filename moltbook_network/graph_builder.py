import networkx as nx
import pandas as pd
from tqdm import tqdm


def build_reply_graph(df):

    print("Building reply graph...")
    G = nx.DiGraph()

    #  comment_id -> agent_id
    comment_to_agent = dict(zip(df["comment_id"], df["agent_id"]))

    #  agent_id -> REAL agent name (agent_name_y)
    agent_id_to_name = dict(zip(df["agent_id"], df["agent_name_y"]))

    for _, row in tqdm(df.iterrows(), total=len(df)):

        parent_comment_id = row["parent_id"]
        child_agent_id = row["agent_id"]

        if pd.isna(parent_comment_id):
            continue

        if parent_comment_id not in comment_to_agent:
            continue

        parent_agent_id = comment_to_agent[parent_comment_id]

        parent_name = agent_id_to_name.get(parent_agent_id)
        child_name = agent_id_to_name.get(child_agent_id)

        if (
            pd.notna(parent_name)
            and pd.notna(child_name)
            and parent_name != child_name
        ):
            G.add_edge(child_name, parent_name)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    return G


def get_largest_component(G):

    if G.number_of_nodes() == 0:
        return G

    largest = max(nx.weakly_connected_components(G), key=len)
    G_main = G.subgraph(largest).copy()

    print("Largest component size:", G_main.number_of_nodes())
    return G_main
