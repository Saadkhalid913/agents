#!/usr/bin/env bash
# Thin wrapper — delegates to build.py
exec python3 "$(dirname "${BASH_SOURCE[0]}")/build.py" "$@"
