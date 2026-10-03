#!/usr/bin/env python3
"""Local pass-through proxy to the xAI API that records token usage and enforces a spend cap.

    python scripts/xai_usage_proxy.py --port 5097 --budget-usd 10 --log backend/logs/xai_usage.jsonl
    # then LLM_BASE_URL=http://127.0.0.1:5097/v1 (and LLM_BOOST_BASE_URL)

Every JSON response's `usage` is appended to the log with an estimated cost.
Once the estimated total passes --budget-usd, further requests get HTTP 429.
The Authorization header is forwarded untouched and never logged.
Non-streaming requests only (MiroFish/OASIS don't stream).
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# USD per 1M tokens: (input, cached input, output). From docs.x.ai/docs/models (Oct 2026).
PRICES = {
    "grok-4.3": (1.25, 0.20, 2.50),
    "grok-4.20-0309-non-reasoning": (1.25, 0.20, 2.50),
    "grok-4.20-0309-reasoning": (1.25, 0.20, 2.50),
    "grok-4.7": (2.00, 0.50, 6.00),
    "grok-4.6": (2.00, 0.50, 6.00),
    "grok-4.5": (2.00, 0.30, 6.00),
}
DEFAULT_PRICE = (2.00, 0.50, 6.00)
LOCK = threading.Lock()
TOTAL = {"calls": 0, "prompt": 0, "cached": 0, "completion": 0, "reasoning": 0, "usd": 0.0, "by_model": {}}


def cost(model, usage):
    p_in, p_cached, p_out = PRICES.get(model, DEFAULT_PRICE)
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    # xAI reports reasoning tokens separately from completion_tokens; both bill as output.
    usd = ((prompt - cached) * p_in + cached * p_cached + (completion + reasoning) * p_out) / 1e6
    if usage.get("cost_in_usd_ticks"):  # xAI's billed cost; 1 tick = 1e-10 USD
        usd = usage["cost_in_usd_ticks"] / 1e10
    return prompt, cached, completion, reasoning, usd


class Handler(BaseHTTPRequestHandler):
    upstream = "https://api.x.ai"
    budget = 10.0
    log_path = None

    def log_message(self, *a):
        pass

    def _reply(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _forward(self, method):
        if self.path == "/__usage":
            with LOCK:
                return self._reply(200, json.dumps(TOTAL).encode())
        with LOCK:
            over = TOTAL["usd"] >= self.budget
        if over and method == "POST":
            msg = {"error": {"message": f"local budget cap ${self.budget} reached", "type": "budget_exceeded"}}
            return self._reply(429, json.dumps(msg).encode())
        length = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "accept-encoding", "connection")}
        req = urllib.request.Request(self.upstream + self.path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                code, body, ctype = resp.status, resp.read(), resp.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as err:
            code, body, ctype = err.code, err.read(), err.headers.get("Content-Type", "application/json")
        except Exception as err:  # network failure -> 502 so clients retry
            return self._reply(502, json.dumps({"error": {"message": str(err)}}).encode())
        try:
            obj = json.loads(body)
            usage = obj.get("usage") if isinstance(obj, dict) else None
        except Exception:
            usage = None
        if usage:
            model = obj.get("model") or ""
            prompt, cached, completion, reasoning, usd = cost(model, usage)
            with LOCK:
                TOTAL["calls"] += 1
                TOTAL["prompt"] += prompt
                TOTAL["cached"] += cached
                TOTAL["completion"] += completion
                TOTAL["reasoning"] += reasoning
                TOTAL["usd"] += usd
                bm = TOTAL["by_model"].setdefault(model, {"calls": 0, "usd": 0.0})
                bm["calls"] += 1
                bm["usd"] += usd
                if self.log_path:
                    with open(self.log_path, "a") as fh:
                        fh.write(json.dumps({"t": time.time(), "model": model, "status": code, "usage": usage, "usd": usd}) + "\n")
        self._reply(code, body, ctype)

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5097)
    ap.add_argument("--budget-usd", type=float, default=10.0)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    Handler.budget, Handler.log_path = a.budget_usd, a.log
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
