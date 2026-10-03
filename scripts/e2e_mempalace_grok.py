#!/usr/bin/env python3
"""End-to-end smoke run against a running MiroFish backend (MemPalace + Grok).

    scripts/mempalace_grok_backend.sh &          # terminal 1
    backend/.venv/bin/python scripts/e2e_mempalace_grok.py --rounds 12   # terminal 2

Steps: ontology/generate -> graph/build -> simulation/create -> prepare
(personas + config) -> start (max_rounds) -> report/generate. Only the
standard library is used. Each call's result is printed; the final report is
saved next to this run's summary JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = ROOT / "examples/mempalace_grok/sample_seed_product_launch.md"
REQUIREMENT = (
    "SAMPLE run: simulate how social media reacts over the first days after the fictional "
    "Lumora Labs Halo Kettle launch, focusing on the Halo Plus subscription, competitor "
    "criticism, consumer-advocate scrutiny and the risk of a launch delay."
)


def call(base: str, method: str, path: str, payload=None, *, files=None, form=None, timeout=600, soft=False):
    url = base.rstrip("/") + path
    headers = {}
    body = None
    if files is not None:
        boundary = uuid.uuid4().hex
        parts = []
        for key, value in (form or {}).items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
        for key, (name, data, ctype) in files.items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{name}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n"
            )
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as err:
        out = json.loads(err.read().decode() or "{}")
    if not out.get("success", False):
        if soft:
            return {"_error": out}
        raise SystemExit(f"{method} {path} failed: {json.dumps(out, ensure_ascii=False)[:2000]}")
    return out.get("data") or {}


def poll(label: str, fn, done, *, interval=5, timeout=7200):
    start = time.time()
    last = None
    while True:
        data = fn()
        msg = f"{data.get('status') or data.get('runner_status')} {data.get('progress', data.get('progress_percent', ''))} {data.get('message', '')}".strip()
        if msg != last:
            print(f"  [{label} {int(time.time() - start)}s] {msg[:200]}", flush=True)
            last = msg
        verdict = done(data)
        if verdict is True:
            return data
        if verdict is False:
            raise SystemExit(f"{label} failed: {json.dumps(data, ensure_ascii=False)[:2000]}")
        if time.time() - start > timeout:
            raise SystemExit(f"{label} timed out")
        time.sleep(interval)


def task_done(data):
    status = data.get("status")
    if status in ("completed", "ready"):
        return True
    if status == "failed":
        return False
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5001")
    ap.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--platform", default="parallel", choices=["parallel", "twitter", "reddit"])
    ap.add_argument("--out", type=Path, default=ROOT / "backend/uploads/e2e_runs")
    ap.add_argument("--stop-after", choices=["graph", "prepare"], default=None,
                    help="stop early (used for the mocked-LLM smoke test)")
    args = ap.parse_args()
    base = args.base
    t0 = time.time()
    summary: dict = {"seed": str(args.seed), "rounds": args.rounds}

    print("1/6 ontology/generate")
    data = call(base, "POST", "/api/graph/ontology/generate",
                files={"files": (args.seed.name, args.seed.read_bytes(), "text/markdown")},
                form={"simulation_requirement": REQUIREMENT, "project_name": "SAMPLE Halo Kettle"})
    project_id = summary["project_id"] = data["project_id"]
    print(f"  project {project_id}")

    print("2/6 graph/build")
    data = call(base, "POST", "/api/graph/build", {"project_id": project_id})
    task_id = data["task_id"]
    poll("build", lambda: call(base, "GET", f"/api/graph/task/{task_id}"), task_done)
    project = call(base, "GET", f"/api/graph/project/{project_id}")
    graph_id = summary["graph_id"] = project["graph_id"]
    graph = call(base, "GET", f"/api/graph/data/{graph_id}")
    summary["graph_nodes"] = graph.get("node_count", len(graph.get("nodes", [])))
    summary["graph_edges"] = graph.get("edge_count", len(graph.get("edges", [])))
    print(f"  graph {graph_id}: {summary['graph_nodes']} nodes, {summary['graph_edges']} edges")

    if args.stop_after == "graph":
        print(json.dumps(summary, indent=2))
        return

    print("3/6 simulation/create")
    data = call(base, "POST", "/api/simulation/create", {"project_id": project_id, "graph_id": graph_id})
    sim_id = summary["simulation_id"] = data["simulation_id"]

    print("4/6 simulation/prepare (personas + config)")
    data = call(base, "POST", "/api/simulation/prepare", {"simulation_id": sim_id, "parallel_profile_count": 4})
    if data.get("status") != "ready":
        prep_task = data.get("task_id")
        poll("prepare", lambda: call(base, "POST", "/api/simulation/prepare/status",
                                     {"task_id": prep_task, "simulation_id": sim_id}), task_done)
    profiles = call(base, "GET", f"/api/simulation/{sim_id}/profiles")
    summary["profiles"] = profiles.get("count", len(profiles.get("profiles", [])))
    print(f"  personas: {summary['profiles']}")

    if args.stop_after == "prepare":
        print(json.dumps(summary, indent=2))
        return

    print(f"5/6 simulation/start ({args.rounds} rounds, graph memory update on)")
    call(base, "POST", "/api/simulation/start", {
        "simulation_id": sim_id, "platform": args.platform, "max_rounds": args.rounds,
        "enable_graph_memory_update": True, "force": True,
    })

    def sim_done(d):
        status = d.get("runner_status")
        if status in ("completed", "stopped"):
            return True
        # The runner stays "running" (waiting for interview commands) after the
        # rounds finish; treat both platforms completing as done, like the UI.
        wanted = {"parallel": ("twitter_completed", "reddit_completed"),
                  "twitter": ("twitter_completed",), "reddit": ("reddit_completed",)}[args.platform]
        if all(d.get(k) for k in wanted):
            return True
        if status == "failed":
            return False
        return None

    run = poll("simulation", lambda: call(base, "GET", f"/api/simulation/{sim_id}/run-status"), sim_done, interval=10)
    summary["actions"] = run.get("total_actions_count")
    summary["rounds_done"] = run.get("current_round")

    # Stop the runner (it idles in interview mode) so graph-memory ingestion
    # can drain; report generation waits for a terminal run status.
    call(base, "POST", "/api/simulation/stop", {"simulation_id": sim_id}, soft=True)

    print("6/6 report/generate")
    for _ in range(120):
        data = call(base, "POST", "/api/report/generate", {"simulation_id": sim_id}, soft=True)
        err = data.get("_error")
        if not err:
            break
        if "still active" not in str(err.get("error", "")):
            raise SystemExit(f"report/generate failed: {err}")
        print("  waiting for graph-memory ingestion to drain...", flush=True)
        time.sleep(10)
    else:
        raise SystemExit("report/generate: ingestion never drained")
    rep_task = data.get("task_id")
    poll("report", lambda: call(base, "POST", "/api/report/generate/status",
                                {"task_id": rep_task, "simulation_id": sim_id}), task_done, interval=10)
    report = call(base, "GET", f"/api/report/by-simulation/{sim_id}")
    report_id = summary["report_id"] = report.get("report_id")
    args.out.mkdir(parents=True, exist_ok=True)
    md = report.get("markdown_content") or ""
    report_path = args.out / f"{sim_id}_report.md"
    report_path.write_text(md, encoding="utf-8")
    summary["report_path"] = str(report_path)
    summary["elapsed_s"] = round(time.time() - t0)
    (args.out / f"{sim_id}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
