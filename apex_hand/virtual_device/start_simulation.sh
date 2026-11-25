#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "用法: $0 [left|right|dual] [--headless]"
}

if [[ $# -gt 0 && "$1" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
  side="$1"
  shift
else
  side="right"
fi

case "$side" in
  left|right|dual) ;;
  *) usage >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u "$script_dir/mujoco_server.py" --side "$side" "$@"
