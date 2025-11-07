from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
def load_flan(name="google/flan-t5-base"):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
    return tok, mdl
def _gen(mdl, tok, prompt, max_new_tokens=128):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=True)
def zero_shot(m,t,p): return _gen(m,t,p)
def few_shot(m,t,ex,p): return _gen(m,t,"\n".join(ex)+f"\n{p}")
def cot(m,t,p): return _gen(m,t,p+"\nLet's think step by step.")
