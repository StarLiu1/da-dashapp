#!/usr/bin/env bash

# Install dependencies and use a cached virtual environment
if [[ -d .venv ]]; then
  echo "Reusing existing virtual environment"
  source .venv/bin/activate
else
  echo "Creating new virtual environment"
  python -m venv .venv
  source .venv/bin/activate
fi

pip install --upgrade pip
pip install -r requirements.txt
