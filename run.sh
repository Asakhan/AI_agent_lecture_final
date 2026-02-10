#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

exec python main.py
