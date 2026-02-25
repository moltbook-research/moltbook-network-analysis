import networkx as nx
import pandas as pd
from collections import Counter


def print_basic_stats(G):
    print("\nBASIC NETWORK STATS")
    print("Total Nodes:", G.number_of_nodes())
    print("Total Edges:", G.number_of_edges())
    print("Density:", nx.density(G))

    if G.number_of_nodes() > 0:
        print("Average Degree:", sum(dict(G.degree()).values()) / G.number_of_nodes())


def print_top_degree(G, top_n=10):
    print("\nTOP HUB AGENTS (BY TOTAL DEGREE)")
    degree_dict = dict(G.degree())
    top = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    for i, (node, degree) in enumerate(top, 1):
        print(f"{i}. {node} — {degree} connections")

    print("\nInterpretation:")
    print("These agents act as central hubs in the reply network.")


def print_top_in_degree(G, top_n=10):
    print("\nMOST REPLIED-TO AGENTS (IN-DEGREE)")
    in_degree_dict = dict(G.in_degree())
    top = sorted(in_degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    for i, (node, degree) in enumerate(top, 1):
        print(f"{i}. {node} — received {degree} replies")

    print("\nInterpretation:")
    print("These agents attract the most responses and function as conversation magnets.")


def print_top_out_degree(G, top_n=10):
    print("\nMOST ACTIVE REPLYING AGENTS (OUT-DEGREE)")
    out_degree_dict = dict(G.out_degree())
    top = sorted(out_degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    for i, (node, degree) in enumerate(top, 1):
        print(f"{i}. {node} — replied {degree} times")

    print("\nInterpretation:")
    print("These agents are the most active participants in discussions.")


def analyze_communities(G, communities):
    print("\nCOMMUNITY ANALYSIS")

    for i, community in enumerate(communities):
        subgraph = G.subgraph(community)
        degrees = dict(subgraph.degree())

        if len(degrees) == 0:
            continue

        leader = max(degrees, key=degrees.get)

        print(f"\nCommunity {i}:")
        print("Size:", len(community))
        print("Central Agent:", leader)
        print("Connections of Leader:", degrees[leader])

    print("\nInterpretation:")
    print("Communities are structurally organized around local central agents.")
