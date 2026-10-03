#!/usr/bin/env bash
# Start the MiroFish backend with MemPalace graph memory and xAI Grok.
# Requires XAI_API_KEY in the environment; never written to disk.
set -euo pipefail
cd "$(dirname "$0")/../backend"
: "${XAI_API_KEY:?XAI_API_KEY is not set}"
export LLM_API_KEY="$XAI_API_KEY"
export LLM_BOOST_API_KEY="$XAI_API_KEY"
export GRAPH_MEMORY_BACKEND="${GRAPH_MEMORY_BACKEND:-mempalace}"
export FLASK_HOST="${FLASK_HOST:-127.0.0.1}"
export FLASK_PORT="${FLASK_PORT:-5001}"
exec uv run python run.py
