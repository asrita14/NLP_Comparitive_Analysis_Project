from .baselines import _gen
def sge_infer(m,t,task,k=3):
    meta = f"Generate {k} input–output examples for this task.\nTask:\n{task}\nExamples:\n"
    examples = _gen(m,t,meta,256)
    final = examples + "\nNow answer:\n" + task
    return _gen(m,t,final)
