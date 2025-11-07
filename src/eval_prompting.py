# src/eval_prompting.py
import json, argparse
from pathlib import Path
from src.baselines import load_flan, _gen


LABELS = ["World", "Sports", "Business", "Sci/Tech"]

SYSTEM = (
"Task: Classify a news article into one of the following labels: "
+ ", ".join(LABELS) + ". Return only the label.\n"
)

PROMPT_TMPL = (
"{system}\n"
"Text: {text}\n"
"Answer:"
)

def predict_one(mdl, tok, text, max_new_tokens=16):
    prompt = PROMPT_TMPL.format(system=SYSTEM, text=text.strip())
    out = _gen(mdl, tok, prompt, max_new_tokens=max_new_tokens).strip()
    # simple post-process: pick first label mentioned; default to World
    out_norm = next((lab for lab in LABELS if lab.lower() in out.lower()), "World")
    return out_norm

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main(data_dir="data/ag_news_small", k_eval=200, model_name="google/flan-t5-base"):
    tok, mdl = load_flan(model_name)
    # tiny eval for speed
    test_path = Path(data_dir) / "test.jsonl"
    rows = list(load_jsonl(test_path))[:k_eval]

    gold = [LABELS[r["label"]] for r in rows]
    pred = [predict_one(mdl, tok, r["text"]) for r in rows]
    acc = sum(g==p for g,p in zip(gold,pred)) / len(rows)
    print(f"Eval N={len(rows)}  Accuracy={acc:.3f}")

    Path("results/plots").mkdir(parents=True, exist_ok=True)
    with open("results/prompting_metrics.json", "w") as f:
        json.dump({"dataset":"ag_news", "N":len(rows), "accuracy":acc, "model":model_name}, f, indent=2)
    print("Saved results to results/prompting_metrics.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/ag_news_small")
    ap.add_argument("--k_eval", type=int, default=200)
    ap.add_argument("--model", default="google/flan-t5-base")
    args = ap.parse_args()
    main(args.data_dir, args.k_eval, args.model)
