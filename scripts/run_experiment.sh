#!/usr/bin/env bash
# Convenience wrapper around `python main.py`.
#
# Usage:
#   ./scripts/run_experiment.sh                    # full 75-fold sweep
#   ./scripts/run_experiment.sh --quick            # 2-fold smoke test
#   ./scripts/run_experiment.sh --models codebert  # subset
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -o allexport
  # shellcheck disable=SC1091
  source .env
  set +o allexport
fi

exec python main.py "$@"
