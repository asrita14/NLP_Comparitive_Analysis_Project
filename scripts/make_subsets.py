# scripts/make_subsets.py
from datasets import load_dataset
from pathlib import Path
import json

def dump_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main(n_train=1000, n_test=500, cache_dir=None):
    ds = load_dataset("ag_news", cache_dir=cache_dir)
    train = ds["train"].select(range(min(n_train, len(ds["train"]))))
    test  = ds["test"].select(range(min(n_test,  len(ds["test"]))))

    # Normalize to {text, label}
    train_rows = [{"text": r["text"], "label": int(r["label"])} for r in train]
    test_rows  = [{"text": r["text"], "label": int(r["label"])} for r in test]

    out_dir = Path("data/ag_news_small")
    dump_jsonl(train_rows, out_dir / "train.jsonl")
    dump_jsonl(test_rows,  out_dir / "test.jsonl")
    print("Wrote:", out_dir / "train.jsonl", out_dir / "test.jsonl")

if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=500)
    p.add_argument("--cache_dir", type=str, default=None)
    args = p.parse_args()
    main(args.n_train, args.n_test, args.cache_dir)
