#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def done_if_all_exist(paths: List[str], cwd: Path) -> bool:
    if not paths:
        return False
    return all((cwd / p).exists() for p in paths)


def load_state(path: Path) -> Dict[str, Any]:
    if path.exists():
        return load_json(path)
    return {"tasks": {}, "history": []}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run_task(task: Dict[str, Any], cwd: Path, env_extra: Dict[str, str], log_dir: Path, dry_run: bool) -> int:
    task_id = str(task["id"])
    cmd = str(task["cmd"])
    shell = bool(task.get("shell", True))
    task_env = dict(env_extra)
    task_env.update({str(k): str(v) for k, v in task.get("env", {}).items()})
    env = os.environ.copy()
    env.update(task_env)

    ensure_dir(log_dir)
    out_log = log_dir / f"{task_id}.out.log"
    err_log = log_dir / f"{task_id}.err.log"

    print(f"[campaign] run {task_id}: {cmd}")
    if dry_run:
        return 0

    with out_log.open("ab") as fout, err_log.open("ab") as ferr:
        fout.write(f"\n=== {now_ts()} START {task_id} ===\n".encode())
        ferr.write(f"\n=== {now_ts()} START {task_id} ===\n".encode())
        proc = subprocess.run(
            cmd if shell else shlex.split(cmd),
            cwd=str(cwd),
            env=env,
            shell=shell,
            stdout=fout,
            stderr=ferr,
            check=False,
        )
        fout.write(f"\n=== {now_ts()} END {task_id} rc={proc.returncode} ===\n".encode())
        ferr.write(f"\n=== {now_ts()} END {task_id} rc={proc.returncode} ===\n".encode())
    return int(proc.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description="Resumable campaign runner for A100 experiment queues")
    p.add_argument("--campaign", required=True, help="Path to campaign JSON spec")
    p.add_argument("--only", default=None, help="Comma-separated task ids to run")
    p.add_argument("--from-task", default=None, help="Start from this task id (inclusive)")
    p.add_argument("--dry-run", action="store_true", help="Print tasks, do not execute")
    p.add_argument("--force", action="store_true", help="Run even if done_if_all_exist is satisfied")
    p.add_argument("--continue-on-error", action="store_true", help="Do not stop on task failure")
    args = p.parse_args()

    campaign_path = Path(args.campaign).resolve()
    spec = load_json(campaign_path)
    cwd = (campaign_path.parent / spec.get("workdir", ".")).resolve()
    log_dir = (cwd / spec.get("log_dir", f"artifacts_campaigns/{campaign_path.stem}")).resolve()
    state_path = log_dir / "state.json"
    state = load_state(state_path)

    tasks: List[Dict[str, Any]] = list(spec.get("tasks", []))
    if not tasks:
        raise ValueError("Campaign has no tasks")

    only = None
    if args.only:
        only = {x.strip() for x in str(args.only).split(",") if x.strip()}

    if args.from_task:
        ids = [str(t["id"]) for t in tasks]
        if args.from_task not in ids:
            raise ValueError(f"from-task not found: {args.from_task}")
        start_idx = ids.index(args.from_task)
        tasks = tasks[start_idx:]

    env_extra = {str(k): str(v) for k, v in spec.get("env", {}).items()}

    print(f"[campaign] name={spec.get('name', campaign_path.stem)} cwd={cwd}")
    print(f"[campaign] log_dir={log_dir}")
    if only:
        print(f"[campaign] only={sorted(only)}")

    for task in tasks:
        task_id = str(task["id"])
        if only and task_id not in only:
            continue

        done_paths = [str(x) for x in task.get("done_if_all_exist", [])]
        if (not args.force) and done_if_all_exist(done_paths, cwd):
            print(f"[campaign] skip {task_id} (done_if_all_exist)")
            state["tasks"][task_id] = {
                "status": "skipped_done",
                "ts_utc": now_ts(),
                "done_if_all_exist": done_paths,
            }
            save_state(state_path, state)
            continue

        rc = run_task(task, cwd=cwd, env_extra=env_extra, log_dir=log_dir, dry_run=bool(args.dry_run))
        rec = {
            "task_id": task_id,
            "ts_utc": now_ts(),
            "returncode": rc,
            "status": "ok" if rc == 0 else "failed",
            "done_if_all_exist": done_paths,
        }
        state["tasks"][task_id] = rec
        state["history"].append(rec)
        save_state(state_path, state)

        if rc != 0 and not args.continue_on_error:
            print(f"[campaign] stop on failure: {task_id} rc={rc}")
            sys.exit(rc)

    print("[campaign] completed")


if __name__ == "__main__":
    main()
