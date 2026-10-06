#!/usr/bin/env bash
# Mocked-live smoke: exercises OpenAI adapter via wiremock-free local hyper? 
# Uses `cargo test` mocked_live_contract + fixture smoke as the offline gate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Fixture offline smoke"
./scripts/application_contract_smoke.sh

echo "==> Mocked-live adapter contract tests"
cargo test -p arsenic-adapters --test mocked_live_contract -- --nocapture
cargo test -p arsenic-core --test stale_and_changed -- --nocapture

echo ""
echo "MOCKED-LIVE SMOKE OK"
