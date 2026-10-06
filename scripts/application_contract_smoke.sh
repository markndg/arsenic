#!/usr/bin/env bash
# Offline Application Contract smoke test — deterministic, no network.
# Covers: PASS, BLOCK, STALE, EXECUTION FAILURE, validated repair.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEMO="$ROOT/examples/customer-support"
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/arsenic-contract-smoke.XXXXXX")"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

echo "==> Building arsenic"
cargo build -q -p arsenic --manifest-path "$ROOT/Cargo.toml"
TARGET_DIR="$(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
BIN="$TARGET_DIR/debug/arsenic"
test -x "$BIN"

cp -R "$DEMO/." "$WORKDIR/"
cd "$WORKDIR"

echo "==> arsenic init"
"$BIN" init --name "Customer Support" --contract "$WORKDIR/contract.json" --project "$WORKDIR"

echo "==> arsenic baseline openai:gpt-current"
"$BIN" baseline openai:gpt-current \
  --project "$WORKDIR" \
  --from-fixtures "$WORKDIR/fixtures/production.json"

echo "==> arsenic qualify (multi-candidate mixed → expect BLOCK exit 1)"
set +e
"$BIN" qualify \
  openai:gpt-safe anthropic:claude-y google:gemini-z \
  --project "$WORKDIR" \
  --from-fixtures "$WORKDIR/fixtures/candidates.json"
QUAL_RC=$?
set -e
if [[ "$QUAL_RC" -ne 1 ]]; then
  echo "expected CI-style block exit 1 from mixed qualify, got $QUAL_RC" >&2
  exit 1
fi

echo "==> PASS path: qualify --ci openai:gpt-safe (expect 0 + SAFE TO MIGRATE)"
set +e
"$BIN" qualify --ci openai:gpt-safe \
  --project "$WORKDIR" \
  --from-fixtures "$WORKDIR/fixtures/candidates.json" >"$WORKDIR/qual-safe.json"
SAFE_RC=$?
set -e
[[ "$SAFE_RC" -eq 0 ]] || { echo "safe candidate should exit 0, got $SAFE_RC" >&2; exit 1; }
python3 - <<PY
import json
d=json.load(open("$WORKDIR/qual-safe.json"))
assert d["aggregate_migration_recommendation_text"]=="SAFE TO MIGRATE", d
assert d["ci_exit_code"]==0
assert d["effective"][0]["evidence_validity"]=="VALID"
assert d["effective"][0]["is_stale"] is False
print("PASS path OK")
PY

echo "==> BLOCK path: qualify --ci google:gemini-z (expect 1 + MIGRATION BLOCKED)"
set +e
"$BIN" qualify --ci google:gemini-z \
  --project "$WORKDIR" \
  --from-fixtures "$WORKDIR/fixtures/candidates.json" >"$WORKDIR/qual-block.json"
BLOCK_RC=$?
set -e
[[ "$BLOCK_RC" -eq 1 ]] || { echo "blocked candidate should exit 1, got $BLOCK_RC" >&2; exit 1; }
python3 - <<PY
import json
d=json.load(open("$WORKDIR/qual-block.json"))
assert d["aggregate_migration_recommendation_text"]=="MIGRATION BLOCKED", d
assert d["ci_exit_code"]==1
print("BLOCK path OK")
PY

echo "==> VALIDATED PATCH: qualify --repair openai:gpt-repairable"
set +e
"$BIN" qualify --repair --ci openai:gpt-repairable \
  --project "$WORKDIR" \
  --from-fixtures "$WORKDIR/fixtures/candidates.json" >"$WORKDIR/qual-repair.json"
REPAIR_RC=$?
set -e
[[ "$REPAIR_RC" -eq 0 ]] || { echo "repaired candidate should exit 0, got $REPAIR_RC" >&2; exit 1; }
python3 - <<PY
import json
d=json.load(open("$WORKDIR/qual-repair.json"))
assert d["ci_exit_code"]==0
assert d["aggregate_migration_recommendation_text"]=="SAFE TO MIGRATE"
rec=d["qualifications"][0]["decision"]
assert rec in ("PASS_WITH_PATCH","PASS","PASS_WITH_WARNINGS"), rec
print("REPAIR path OK:", rec)
PY

QUAL_ID=$(ls "$WORKDIR/.arsenic/qualifications" | grep qual | sort | tail -1 | sed 's/.json//')
echo "==> arsenic patch show $QUAL_ID"
"$BIN" patch show "$QUAL_ID" --project "$WORKDIR" | head -20

echo "==> arsenic impact google:gemini-z"
"$BIN" impact google:gemini-z --project "$WORKDIR" | tee "$WORKDIR/impact-out.txt"
grep -q "MIGRATION BLOCKED" "$WORKDIR/impact-out.txt"

echo "==> arsenic report (HTML + JSON)"
"$BIN" report --project "$WORKDIR" --output "$WORKDIR/application-report.html"
test -f "$WORKDIR/application-report.html"
grep -E "SAFE TO MIGRATE|MIGRATION BLOCKED|REVIEW|STALE|INCOMPLETE" "$WORKDIR/application-report.html" >/dev/null
"$BIN" report --json --project "$WORKDIR" >"$WORKDIR/app-report.json"
python3 - <<PY
import json
d=json.load(open("$WORKDIR/app-report.json"))
assert "aggregate_migration_recommendation_text" in d
# Mixed candidates include a blocker → must not be SAFE overall
assert d["aggregate_migration_recommendation_text"] != "SAFE TO MIGRATE", d["aggregate_migration_recommendation_text"]
print("report aggregate:", d["aggregate_migration_recommendation_text"])
PY

echo "==> EXECUTION FAILURE path: fixture with timeout"
python3 - <<'PY'
import json
from pathlib import Path
# Mirror prompts used by the demo contract so every item hits execution_error.
prompts = ["refund_decision","order_lookup","refund_json","greeting","abuse_request"]
fx={"openai:gpt-timeout": {}}
for p in prompts:
    fx["openai:gpt-timeout"][p]={
      "prompt_id": p,
      "content": "Refunds over £500 require manager approval" if p=="refund_decision" else "",
      "latency_ms": 1,
      "cost_usd": 0.01,
      "tool_calls": [],
      "execution_error": "TIMEOUT",
      "execution_error_message": "provider timed out"
    }
Path("fixtures/timeout.json").write_text(json.dumps(fx, indent=2))
PY
set +e
"$BIN" qualify --ci openai:gpt-timeout \
  --project "$WORKDIR" \
  --from-fixtures "$WORKDIR/fixtures/timeout.json" >"$WORKDIR/qual-timeout.json"
TIMEOUT_RC=$?
set -e
[[ "$TIMEOUT_RC" -ne 0 ]] || { echo "execution failure must be non-zero, got 0" >&2; exit 1; }
python3 - <<PY
import json
d=json.load(open("$WORKDIR/qual-timeout.json"))
assert d["aggregate_migration_recommendation_text"]=="QUALIFICATION INCOMPLETE", d
assert d["ci_exit_code"]==3
assert d["effective"][0]["evidence_validity"]=="INCOMPLETE"
assert "SAFE" not in d["aggregate_migration_recommendation_text"]
print("EXECUTION FAILURE path OK")
PY

echo "==> STALE path: bump contract, report/impact without requalify"
python3 - <<'PY'
import json
from pathlib import Path
p=Path(".arsenic/contract/contract.json")
c=json.loads(p.read_text())
c["version"]=int(c.get("version",1))+1
if c.get("items"):
    c["items"][0].setdefault("tags", []).append("smoke-stale-bump")
p.write_text(json.dumps(c, indent=2))
PY
"$BIN" report --json --project "$WORKDIR" >"$WORKDIR/app-report-stale.json"
"$BIN" impact openai:gpt-safe --project "$WORKDIR" | tee "$WORKDIR/impact-stale.txt"
python3 - <<PY
import json
d=json.load(open("$WORKDIR/app-report-stale.json"))
text=d["aggregate_migration_recommendation_text"]
assert text=="STALE — REQUALIFY REQUIRED", text
assert "SAFE TO MIGRATE" not in text
for q in d["qualifications"]:
    assert q["is_stale"] is True, q
    assert q["migration_recommendation_text"]!="SAFE TO MIGRATE"
print("STALE path OK:", text)
PY
grep -q "STALE" "$WORKDIR/impact-stale.txt"
if grep -q "SAFE TO MIGRATE" "$WORKDIR/impact-stale.txt"; then
  echo "stale impact must not say SAFE TO MIGRATE" >&2
  exit 1
fi

echo "==> arsenic contract diff"
"$BIN" contract diff --project "$WORKDIR" || true

echo "==> deterministic replay check (restore contract for live qualify)"
# Restore contract so fixtures still match requirement set for replay determinism check
cp "$DEMO/contract.json" "$WORKDIR/.arsenic/contract/contract.json"
"$BIN" qualify --json openai:gpt-safe \
  --project "$WORKDIR" \
  --from-fixtures "$WORKDIR/fixtures/candidates.json" > "$WORKDIR/replay-a.json"
"$BIN" qualify --json openai:gpt-safe \
  --project "$WORKDIR" \
  --from-fixtures "$WORKDIR/fixtures/candidates.json" > "$WORKDIR/replay-b.json"
python3 - <<'PY'
import json
a=json.load(open("replay-a.json"))
b=json.load(open("replay-b.json"))
qa=a["qualifications"][0]
qb=b["qualifications"][0]
assert qa["decision"]==qb["decision"]
assert [r["outcome"] for r in qa["item_results"]]==[r["outcome"] for r in qb["item_results"]]
assert a["aggregate_migration_recommendation_text"]==b["aggregate_migration_recommendation_text"]
print("replay ok:", qa["decision"])
PY

echo ""
echo "SMOKE OK — Application Contracts offline path (PASS/BLOCK/STALE/INCOMPLETE/REPAIR)"
echo "workdir was $WORKDIR (cleaned on exit)"
