# Pre-flight for pilot 2, check 2: prompt variants on pilot seeds 1000-1001, base weights, greedy, 96 tokens.
# Choice rule fixed before the run: most no_skill root successes, ties by in-context (topk + full) successes.
import glob, json, os, shutil, subprocess, sys
WORK = "/kaggle/working/skill-library"
if not os.path.exists(WORK):
    src = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/PROTOCOL_sweep.md", recursive=True)][0]
    shutil.copytree(src, WORK)
sys.path.insert(0, WORK); os.chdir(WORK)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
from sweep.config import load_sweep_config
from sweep.backends import HFSweepBackend, Query
from sweep import prompts as P
from env.procgen import generate_world, run_program
from agent.lm_policy import extract_program

EX2 = ("Example 2 (a different world, no known skills):\nKnown skills:\n(none)\nTask: Collect 3 wood from the forest.\n"
       "Program:\ngoto(forest)\ngather(wood, 3)\n")
OLD_RULE = "Do not call {n} itself; write its steps (other skills may be called).\n"
NEW_RULE = ("Rules: call() only skills listed under Known skills, and never call {n}. "
            "Answer with the program only: DSL instructions, one per line, no other text.\n")
UNNAMED = ("Rules: call() only skills listed under Known skills, and do not call the skill for this task itself. "
           "Answer with the program only: DSL instructions, one per line, no other text.\n")

def make(rule, ex2):
    def bp(world, arm, task, lib, k):
        return (f"{world.dsl_reference()}\n{P.FORMAT_EXAMPLE}\n" + (EX2 + "\n" if ex2 else "") +
                f"Known skills:\n{P.skills_block(arm, task, lib, k)}\nTask: {task.description}\n"
                f"Goal: obtain {task.task.goal_text()}\n" + rule.format(n=task.name) + "Program:\n")
    return bp

VARIANTS = {
    "A_old_raw": (make(OLD_RULE, False), "raw"),
    "B_new_chat": (make(NEW_RULE, False), "chat"),
    "C_new_ex2_chat": (make(NEW_RULE, True), "chat"),
    "D_new_ex2_raw": (make(NEW_RULE, True), "raw"),
    "E_unnamed_ex2_chat": (make(UNNAMED, True), "chat"),
    "F_unnamed_ex2_raw": (make(UNNAMED, True), "raw"),
}
cfg = load_sweep_config("configs/sweep_pilot.yaml")
be = HFSweepBackend(cfg, cfg.models[0], 1000)
out = []
for vname, (bp, wrapper) in VARIANTS.items():
    cfg.prompt_wrapper = wrapper
    for seed in (1000, 1001):
        w = generate_world(50, seed)
        lib = w.prefix(50)
        comp = [s for s in lib if not s.is_root]
        cases = ([("none", s, []) for s in w.prefix(10) if s.is_root] + [("topk", s, lib) for s in comp[:8]]
                 + [("full", s, lib) for s in comp[8:16]])
        qs = [Query(be.wrap(bp(w, a, s, l, cfg.topk)), s, None, P.in_context_names(a, s, l, cfg.topk)) for a, s, l in cases]
        for (a, s, l), g in zip(cases, be.generate(qs)):
            prog = extract_program(g.text)
            try:
                ok = bool(s.task.check(run_program(w, prog or "goto(nowhere)", [x.name for x in l], banned=(s.name,))))
            except Exception:
                ok = False
            out.append({"variant": vname, "seed": seed, "arm": a, "skill": s.name, "ok": ok, "text": g.text, "gold": s.program})
    r = {a: [o["ok"] for o in out if o["variant"] == vname and o["arm"] == a] for a in ("none", "topk", "full")}
    print(f"SUMMARY {vname}: " + "  ".join(f"{a} {sum(v)}/{len(v)}" for a, v in r.items()), flush=True)
json.dump(out, open("/kaggle/working/prompt_check.json", "w"), indent=1)
