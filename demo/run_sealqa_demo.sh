#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export SEALQA_TRAIN_SIZE="${SEALQA_TRAIN_SIZE:-6}"
export SEALQA_VAL_SIZE="${SEALQA_VAL_SIZE:-2}"
export SEALQA_TEST_SIZE="${SEALQA_TEST_SIZE:-2}"
export SEALQA_ASO_MAX_ITERATIONS="${SEALQA_ASO_MAX_ITERATIONS:-2}"

run_lifecycle() {
  echo "Running SealQA lifecycle demo (root -> generate -> evolve -> prune -> merge)"
  python -m treeskill
}

run_aso() {
  echo "Running SealQA ASO demo (minimal frontier + beam)"
  python -m treeskill sealqa-aso
}

case "${1:-}" in
  --aso)
    run_aso
    ;;
  --lifecycle)
    run_lifecycle
    ;;
  --both)
    run_lifecycle
    run_aso
    ;;
  "")
    run_lifecycle
    ;;
  *)
    echo "Usage: $0 [--lifecycle|--aso|--both]"
    echo "Default is --lifecycle."
    echo "Environment variables: SEALQA_TRAIN_SIZE (default 6), SEALQA_VAL_SIZE (default 2), SEALQA_TEST_SIZE (default 2), SEALQA_ASO_MAX_ITERATIONS (default 2)."
    exit 1
    ;;
esac
