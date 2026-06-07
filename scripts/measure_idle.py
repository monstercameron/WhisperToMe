"""Measure idle CPU / energy proxy for a running WhisperToMe instance.

Energy on a fanless ARM laptop tracks two things: how much CPU time the app burns, and how
often it wakes the cores out of deep idle (context switches). This samples both across the whole
WhisperToMe process tree (desktop host + child voice loop) while it sits idle/listening.

Usage:
    python scripts/measure_idle.py [--seconds 20] [--pid PID] [--label before]

With no --pid it auto-discovers processes whose command line mentions "whispertome". Run it twice
(e.g. --label before / --label after) to A/B a change. Lower CPU% and lower ctx-switches/sec = less
energy. Reports are also appended to artifacts/idle_measurements.jsonl for later comparison.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def discover_pids() -> list[int]:
    """Match the actual app entrypoints (desktop host / voice loop), not every python process
    that merely runs from the whispertome directory (e.g. this script, pytest)."""
    pids: list[int] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        low = cmdline.lower()
        if "measure_idle" in low or "pytest" in low:
            continue
        is_app = (
            "-m whispertome" in low
            or "whispertome.exe" in low
            or ("whispertome" in low and (" desktop" in low or low.rstrip().endswith(" run") or " run " in low))
        )
        if is_app:
            pids.append(proc.info["pid"])
    return pids


def gather_tree(pids: list[int]) -> list[psutil.Process]:
    procs: dict[int, psutil.Process] = {}
    for pid in pids:
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            continue
        procs[pid] = parent
        for child in parent.children(recursive=True):
            procs[child.pid] = child
    return list(procs.values())


def ctx_switches(proc: psutil.Process) -> int:
    try:
        cs = proc.num_ctx_switches()
        return cs.voluntary + cs.involuntary
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def _label_for(proc: psutil.Process) -> str:
    try:
        cmd = " ".join(proc.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return f"pid {proc.pid}"
    low = cmd.lower()
    if " desktop" in low:
        return "desktop-host"
    if "--project-root" in low and "-m whispertome" in low:
        return "voice-loop"
    return f"pid {proc.pid}"


def measure(procs: list[psutil.Process], seconds: int) -> dict:
    cores = psutil.cpu_count(logical=True) or 1
    # Prime per-process cpu_percent (first call returns 0.0 / establishes a baseline).
    for proc in procs:
        try:
            proc.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    cs_start = {p.pid: ctx_switches(p) for p in procs}
    cpu_per_proc = {p.pid: 0.0 for p in procs}
    t0 = time.perf_counter()

    cpu_samples: list[float] = []
    for _ in range(seconds):
        time.sleep(1.0)
        total = 0.0
        for proc in procs:
            try:
                c = proc.cpu_percent(None)  # % of ONE core since last call
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            total += c
            cpu_per_proc[proc.pid] += c
        cpu_samples.append(total)

    elapsed = time.perf_counter() - t0
    cs_end = {p.pid: ctx_switches(p) for p in procs}
    cs_delta = sum(cs_end.get(pid, v) - v for pid, v in cs_start.items())

    # Per-process breakdown so we know which loop dominates the wakeups.
    by_proc = []
    for proc in procs:
        d = cs_end.get(proc.pid, cs_start[proc.pid]) - cs_start[proc.pid]
        by_proc.append({
            "pid": proc.pid,
            "label": _label_for(proc),
            "cpu_pct_1core_mean": round(cpu_per_proc[proc.pid] / len(cpu_samples), 2),
            "ctx_switches_per_s": round(d / elapsed, 1),
        })
    by_proc.sort(key=lambda r: r["ctx_switches_per_s"], reverse=True)

    sys_cpu = [c / cores for c in cpu_samples]  # normalized to whole-machine %
    return {
        "n_procs": len(procs),
        "cores": cores,
        "seconds": round(elapsed, 1),
        "cpu_pct_1core_mean": round(sum(cpu_samples) / len(cpu_samples), 2),
        "cpu_pct_1core_max": round(max(cpu_samples), 2),
        "cpu_pct_system_mean": round(sum(sys_cpu) / len(sys_cpu), 3),
        "cpu_pct_system_max": round(max(sys_cpu), 3),
        "ctx_switches_per_s": round(cs_delta / elapsed, 1),
        "by_proc": by_proc,
    }


def thread_breakdown(pid: int, seconds: int) -> None:
    """Sample per-thread CPU time deltas for one process to find which loop is busy."""
    proc = psutil.Process(pid)
    before = {t.id: (t.user_time + t.system_time) for t in proc.threads()}
    t0 = time.perf_counter()
    time.sleep(seconds)
    elapsed = time.perf_counter() - t0
    after = {t.id: (t.user_time + t.system_time) for t in proc.threads()}
    rows = []
    for tid, total in after.items():
        delta = total - before.get(tid, total)
        rows.append((tid, delta, 100.0 * delta / elapsed))
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"\n=== per-thread CPU for pid {pid} over {elapsed:.1f}s ===")
    for tid, delta, pct in rows:
        print(f"  thread {tid:<8} cpu_time {delta:6.3f}s  ({pct:5.2f}% of one core)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--pid", type=int, default=None, help="root PID (else auto-discover)")
    ap.add_argument("--label", default="", help="tag for this run (e.g. before/after)")
    ap.add_argument("--threads", type=int, default=None,
                    help="instead of the tree summary, show per-thread CPU for this PID")
    args = ap.parse_args()

    if args.threads is not None:
        thread_breakdown(args.threads, args.seconds)
        return 0

    pids = [args.pid] if args.pid else discover_pids()
    if not pids:
        print("No WhisperToMe processes found. Is the app running?")
        return 1
    procs = gather_tree(pids)
    print(f"Measuring {len(procs)} process(es) for {args.seconds}s "
          f"(roots={pids})... keep the app idle/listening.")
    result = measure(procs, args.seconds)
    result["label"] = args.label
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    print("\n=== idle measurement ===")
    print(f"  label                : {args.label or '(none)'}")
    print(f"  processes / cores    : {result['n_procs']} / {result['cores']}")
    print(f"  CPU% (1 core)  mean  : {result['cpu_pct_1core_mean']}   max: {result['cpu_pct_1core_max']}")
    print(f"  CPU% (system)  mean  : {result['cpu_pct_system_mean']}%  max: {result['cpu_pct_system_max']}%")
    print(f"  ctx switches / sec   : {result['ctx_switches_per_s']}   <- wakeup proxy (lower=better)")
    print("  per-process (by wakeups):")
    for row in result["by_proc"]:
        print(f"    {row['label']:<14} pid {row['pid']:<6} "
              f"cpu {row['cpu_pct_1core_mean']:>5}%  ctx/s {row['ctx_switches_per_s']:>7}")

    out = Path("artifacts/idle_measurements.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")
    print(f"\n  appended to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
