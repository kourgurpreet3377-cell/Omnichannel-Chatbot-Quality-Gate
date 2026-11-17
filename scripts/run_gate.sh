#!/usr/bin/env bash
# The release gate, in order. Each stage must pass before the next (more expensive) one starts.
#   1. API smoke  - deterministic, milliseconds, no tokens
#   2. UI smoke   - deterministic, seconds, no tokens
#   3. LLM evals  - DeepEval judge, minutes, costs tokens
set -euo pipefail

echo "== Stage 1/3: API smoke gate =="
pytest tests/smoke/test_api_smoke.py -q

echo "== Stage 2/3: UI smoke gate =="
pytest tests/smoke/test_ui_smoke.py -q

judge_key="${OPENAI_API_KEY:-}"
if [ "${EVAL_JUDGE_PROVIDER:-openai}" = "anthropic" ]; then judge_key="${ANTHROPIC_API_KEY:-}"; fi
if [ -z "$judge_key" ] && [ -z "${CI:-}" ]; then
  echo "== Stage 3/3: LLM evals SKIPPED =="
  echo "No judge API key found. Set OPENAI_API_KEY (or ANTHROPIC_API_KEY with EVAL_JUDGE_PROVIDER=anthropic) in .env to run it."
  exit 0
fi

echo "== Stage 3/3: LLM evals (channel: ${CHATBOT_CHANNEL:-api}) =="
deepeval test run evals
