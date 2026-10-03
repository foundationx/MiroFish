#!/usr/bin/env python3
"""Tiny OpenAI-compatible mock for offline smoke tests of the MemPalace path.

Answers /v1/chat/completions with canned JSON: an ontology for the ontology
prompt, regex-extracted entities/facts for the MemPalace extraction prompt,
and a generic JSON object otherwise. Not used in production.
"""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ONTOLOGY = {
    "entity_types": [
        {"name": "Company", "description": "A business", "attributes": [{"name": "industry", "type": "text", "description": "industry"}], "examples": ["Lumora Labs"]},
        {"name": "Person", "description": "An individual", "attributes": [{"name": "role", "type": "text", "description": "job role"}], "examples": ["Dana Okafor"]},
        {"name": "Product", "description": "A product", "attributes": [], "examples": ["Halo Kettle"]},
        {"name": "AdvocacyGroup", "description": "Consumer group", "attributes": [], "examples": ["FairTech Alliance"]},
    ],
    "edge_types": [
        {"name": "WORKS_FOR", "description": "employment", "source_targets": [{"source": "Person", "target": "Company"}], "attributes": []},
        {"name": "MAKES", "description": "company makes product", "source_targets": [{"source": "Company", "target": "Product"}], "attributes": []},
        {"name": "CRITICIZES", "description": "criticism", "source_targets": [{"source": "Person", "target": "Product"}, {"source": "AdvocacyGroup", "target": "Company"}], "attributes": []},
    ],
    "analysis_summary": "SAMPLE mock ontology",
}
KNOWN = {
    "Lumora Labs": "Company", "KettleCo": "Company", "BrightCart": "Company", "Northwind Ventures": "Company",
    "Shenwei Precision": "Company", "Dana Okafor": "Person", "Mateo Ruiz": "Person", "Priya Raman": "Person",
    "Elena Brooks": "Person", "Tom Hale": "Person", "Halo Kettle": "Product", "Ember Scale": "Product",
    "FairTech Alliance": "AdvocacyGroup",
}
USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def extraction(text):
    ents = [{"name": n, "type": t, "summary": f"{n} appears in the SAMPLE text.", "attributes": {}} for n, t in KNOWN.items() if n in text]
    names = {e["name"] for e in ents}
    facts = []
    for person, company in [("Dana Okafor", "Lumora Labs"), ("Mateo Ruiz", "Lumora Labs"), ("Priya Raman", "KettleCo"), ("Elena Brooks", "Northwind Ventures")]:
        if person in names and company in names:
            facts.append({"source": person, "relation": "WORKS_FOR", "target": company, "fact": f"{person} works for {company}.", "valid_at": None, "invalid_at": None})
    for prod in ("Halo Kettle", "Ember Scale"):
        if prod in names and "Lumora Labs" in names:
            facts.append({"source": "Lumora Labs", "relation": "MAKES", "target": prod, "fact": f"Lumora Labs makes the {prod}.", "valid_at": "2027-03-03" if prod == "Halo Kettle" else None, "invalid_at": None})
    if "Priya Raman" in names and "Halo Kettle" in names:
        facts.append({"source": "Priya Raman", "relation": "CRITICIZES", "target": "Halo Kettle", "fact": "Priya Raman criticizes the Halo Kettle subscription.", "valid_at": None, "invalid_at": None})
    return {"entities": ents, "facts": facts}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/usage"):
            return self._send(USAGE)
        self._send({"data": [{"id": "mock-grok"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        msgs = req.get("messages", [])
        system = " ".join(m.get("content") or "" for m in msgs if m.get("role") == "system") if msgs else ""
        user = " ".join(m.get("content") or "" for m in msgs if m.get("role") != "system") if msgs else ""
        if "entity_types" in system and "edge_types" in system:
            content = json.dumps(ONTOLOGY)
        elif "extract a knowledge graph" in system:
            content = json.dumps(extraction(user))
        else:
            content = json.dumps({"summary": "mock", "items": []})
        USAGE["calls"] += 1
        USAGE["prompt_tokens"] += len(system + user) // 4
        USAGE["completion_tokens"] += len(content) // 4
        self._send({
            "id": "mock", "object": "chat.completion", "created": 0, "model": req.get("model", "mock"),
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": len(system + user) // 4, "completion_tokens": len(content) // 4, "total_tokens": 0},
        })


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5098
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
