import warnings
warnings.filterwarnings("ignore")

from config import Config, ensure_dirs
from data_io import load_moltbook
from graphs import build_answer_graph, build_discussion_graph, clean_graph
from viz import draw_graph

def run_pipeline() -> None:
    cfg = Config()
    ensure_dirs(cfg)

    print("\nGenerating Moltbook Answer & Discussion Graphs...\n")

    df = load_moltbook()

    # Answer graph pipeline
    G_answer = clean_graph(build_answer_graph(df), cfg)
    draw_graph(G_answer, "answer_person_graph.png", "Moltbook Answer Person Graph", cfg)

    # Discussion graph pipeline
    G_disc = clean_graph(build_discussion_graph(df, cfg), cfg)
    draw_graph(G_disc, "discussion_person_graph.png", "Moltbook Discussion Person Graph", cfg)

    print("\nDone.\n")

if __name__ == "__main__":
    run_pipeline()