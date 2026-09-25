"""Ask one question from the command line: uv run python -m medgraph.ask "..." --mode graphrag"""
import argparse

from medgraph.pipelines import load_engine


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--mode", choices=["vanilla", "graphrag", "cached"], default="graphrag")
    args = p.parse_args()
    engine = load_engine()
    a = getattr(engine, args.mode)(args.question)
    print(a.text)
    print("\n--- provider:", a.provider, "| cache hit:", a.cache_hit, "| timings:",
          {k: round(v) for k, v in a.timings.items()})
    print("--- entities:", a.entities, "| path nodes:", a.path_nodes[:15])
    print("--- citations:", a.citations)


if __name__ == "__main__":
    main()
