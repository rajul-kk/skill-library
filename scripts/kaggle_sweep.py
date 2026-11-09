"""Drive the confirmatory sweep (configs/sweep_reduced.yaml) on Kaggle until every (model, seed) has every arm done.

One round: commit the merged records -> upload the repo (git archive HEAD) as a new version of the Kaggle dataset ->
push notebooks/06_sweep_confirmatory.ipynb as a GPU kernel -> wait -> download its records.jsonl -> merge (union of
rows, earlier rows first) into results/sweep_reduced/records.jsonl -> commit. When nothing is left, run the registered
analysis (scripts/analyze_sweep.py) and commit it.

States, explicit (written to <work>/status.json after every step): uploading, pushing, running, merging, waiting_quota,
done, failed. A round that ends without a new finished arm counts as a failure; two in a row stop the driver rather
than spend GPU quota on a loop. Kaggle's weekly GPU quota shows up as a push or run error mentioning "quota": the driver
then waits (6 h) and tries again, without counting a failure.

usage: python scripts/kaggle_sweep.py --work C:/Users/rajul/.kaggle_sweep --kaggle <path to kaggle.exe>
Needs KAGGLE_API_TOKEN (or ~/.kaggle credentials). Commits only results/sweep_reduced/, only on the sweep branch.
"""
import _paths  # noqa: F401

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sweep.config import load_sweep_config
from sweep.runner import finished_arms

REPO = Path(__file__).resolve().parents[1]
CFG = "configs/sweep_reduced.yaml"
MERGED = REPO / "results" / "sweep_reduced" / "records.jsonl"
BRANCH = "feat/library-size-sweep"
NOTEBOOK = "06_sweep_confirmatory.ipynb"


def log(work: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(work / "sweep.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(work: Path, state: str, **extra) -> None:
    (work / "status.json").write_text(json.dumps({"state": state, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra},
                                                 indent=1), encoding="utf-8")


def remaining(records: Path):
    cfg = load_sweep_config(REPO / CFG)
    keys = {a.key for a in cfg.arm_list()}
    return [(m, s) for m in cfg.models for s in cfg.seeds_for(m) if not keys <= finished_arms(records, m, s)]


def arms_done(records: Path) -> int:
    if not records.exists():
        return 0
    n = 0
    for line in records.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("kind") == "arm_done":
            n += 1
    return n


def merge(into: Path, new: Path) -> int:
    """Union of rows, keeping the existing order and appending unseen rows in their order. Returns rows added."""
    old = into.read_text(encoding="utf-8").splitlines() if into.exists() else []
    seen = set(old)
    add = [l for l in new.read_text(encoding="utf-8").splitlines() if l.strip() and l not in seen]
    into.parent.mkdir(parents=True, exist_ok=True)
    into.write_text("".join(l + "\n" for l in old + add), encoding="utf-8")
    return len(add)


def git(*args, check=True):
    return subprocess.run(["git", "-C", str(REPO), *args], check=check, capture_output=True, text=True)


def commit(paths, message) -> bool:
    if git("branch", "--show-current").stdout.strip() != BRANCH:
        raise SystemExit(f"repo is not on {BRANCH}; not committing")
    paths = [p for p in paths if (REPO / p).exists()]
    if not paths:
        return False
    git("add", "--", *paths)
    if not git("diff", "--cached", "--quiet", "--", *paths, check=False).returncode:
        return False
    git("commit", "-m", message, "--", *paths)
    return True


def kaggle(exe, *args, check=True):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([exe, *args], capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    if check and r.returncode:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {r.stdout[-800:]} {r.stderr[-800:]}")
    return r


def upload_dataset(a, work: Path, round_no: int):
    ds = work / "ds"
    shutil.rmtree(ds / "skill-library", ignore_errors=True)
    (ds / "skill-library").mkdir(parents=True)
    tar = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", "HEAD"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(ds / "skill-library")], input=tar, check=True)
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "skill-library sweep kit", "id": a.dataset,
                                                         "licenses": [{"name": "other"}]}), encoding="utf-8")
    head = git("rev-parse", "--short", "HEAD").stdout.strip()
    kaggle(a.kaggle, "datasets", "version", "-p", str(ds), "-m", f"repo at {head} (confirmatory round {round_no})",
           "--dir-mode", "zip")
    for _ in range(60):                                   # wait until the new version is processed
        r = kaggle(a.kaggle, "datasets", "status", a.dataset, check=False)
        if "ready" in r.stdout.lower():
            return head
        time.sleep(30)
    raise RuntimeError("dataset version not ready after 30 min")


def push_kernel(a, work: Path):
    k = work / "kernel"
    shutil.rmtree(k, ignore_errors=True)
    k.mkdir(parents=True)
    shutil.copyfile(REPO / "notebooks" / NOTEBOOK, k / NOTEBOOK)
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": a.kernel, "title": a.kernel.split("/")[1], "code_file": NOTEBOOK, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [a.dataset], "competition_sources": [], "kernel_sources": []}, indent=1), encoding="utf-8")
    return kaggle(a.kaggle, "kernels", "push", "-p", str(k), check=False)


def wait_kernel(a, work: Path) -> str:
    while True:
        r = kaggle(a.kaggle, "kernels", "status", a.kernel, check=False)
        s = (r.stdout + r.stderr).lower()
        for final in ("complete", "error", "cancel"):
            if f'"{final}' in s or f"status {final}" in s or f".{final}" in s:
                return final
        time.sleep(600)


def fetch_records(a, work: Path, round_no: int) -> Path:
    out = work / f"out_{round_no:02d}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    kaggle(a.kaggle, "kernels", "output", a.kernel, "-p", str(out), check=False)
    found = sorted(out.rglob("runs/sweep_reduced/records.jsonl"))
    return found[0] if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--kaggle", required=True, help="path to the kaggle CLI")
    ap.add_argument("--dataset", default="rajulkabir/skill-library-sweep-kit")
    ap.add_argument("--kernel", default="rajulkabir/skill-library-sweep-confirmatory")
    a = ap.parse_args()
    token = Path.home() / ".kaggle" / "access_token"
    if not os.environ.get("KAGGLE_API_TOKEN") and token.exists():
        os.environ["KAGGLE_API_TOKEN"] = token.read_text(encoding="utf-8").strip()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    failures, round_no = 0, 0
    while True:
        left = remaining(MERGED)
        if not left:
            status(work, "analysing")
            out = REPO / "results" / "sweep_reduced" / "analysis"
            r = subprocess.run([sys.executable, str(REPO / "scripts" / "analyze_sweep.py"), str(MERGED), "--out-dir", str(out)],
                               capture_output=True, text=True)
            (work / "analysis.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
            commit(["results/sweep_reduced"], "results: confirmatory sweep complete (reduced design), registered analysis")
            status(work, "done", arms_done=arms_done(MERGED))
            log(work, "all (model, seed) pairs complete; analysis committed")
            return
        round_no += 1
        before = arms_done(MERGED)
        log(work, f"round {round_no}: {len(left)} (model, seed) pairs left, first {left[0]}; {before} arms done")
        commit(["results/sweep_reduced"], f"results: confirmatory sweep records before Kaggle round {round_no} "
                                          f"({before} arms done, {len(left)} model-seeds left)")
        status(work, "uploading", round=round_no, left=len(left), arms_done=before)
        head = upload_dataset(a, work, round_no)
        status(work, "pushing", round=round_no, commit=head)
        r = push_kernel(a, work)
        text = (r.stdout + r.stderr).lower()
        if r.returncode or "error" in text:
            if "quota" in text:
                status(work, "waiting_quota", round=round_no)
                log(work, f"push refused (quota): {text[-300:]}; waiting 6 h")
                time.sleep(6 * 3600)
                continue
            raise SystemExit(f"kernel push failed: {text[-800:]}")
        log(work, f"kernel pushed for round {round_no} (repo {head})")
        status(work, "running", round=round_no, commit=head, left=len(left))
        final = wait_kernel(a, work)
        log(work, f"kernel finished: {final}")
        status(work, "merging", round=round_no, kernel=final)
        got = fetch_records(a, work, round_no)
        added = merge(MERGED, got) if got else 0
        after = arms_done(MERGED)
        log(work, f"merged {added} new rows; arms done {before} -> {after}")
        commit(["results/sweep_reduced"], f"results: confirmatory sweep, Kaggle round {round_no} "
                                          f"({after} arms done, {len(remaining(MERGED))} model-seeds left)")
        if after > before:
            failures = 0
            continue
        r = kaggle(a.kaggle, "kernels", "output", a.kernel, "-p", str(work / f"out_{round_no:02d}"), check=False)
        logtext = " ".join(p.read_text(encoding="utf-8", errors="replace")[-4000:]
                           for p in (work / f"out_{round_no:02d}").glob("*.log"))
        if "quota" in logtext.lower():
            status(work, "waiting_quota", round=round_no)
            log(work, "no progress and the log mentions quota; waiting 6 h")
            time.sleep(6 * 3600)
            continue
        failures += 1
        log(work, f"round {round_no} made no progress ({final}); failure {failures} of 2")
        if failures >= 2:
            status(work, "failed", round=round_no, kernel=final)
            raise SystemExit("two rounds without progress; stopping (see the kernel log in the out_ folders)")


if __name__ == "__main__":
    main()
