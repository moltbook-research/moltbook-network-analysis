import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    results_dir: str = "answer_discussion_person"
    max_thread_size: int = 40
    min_degree_filter: int = 2

def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.results_dir, exist_ok=True)