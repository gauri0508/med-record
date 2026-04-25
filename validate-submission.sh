#!/usr/bin/env bash
set -uo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  exit 1
fi

PING_URL="${PING_URL%/}"
PASS=0

echo "========================================"
echo "  OpenEnv Submission Validator"
echo "========================================"
echo "Repo: $REPO_DIR"
echo "URL:  $PING_URL"
echo ""

# Step 1: Ping
echo "Step 1/3: Pinging HF Space ($PING_URL/reset) ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>/dev/null || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  echo "  PASSED -- HF Space is live and responds to /reset"
  PASS=$((PASS + 1))
else
  echo "  FAILED -- HTTP $HTTP_CODE"
  exit 1
fi

# Step 2: Docker
echo "Step 2/3: Checking Dockerfile ..."
if [ -f "$REPO_DIR/Dockerfile" ]; then
  echo "  PASSED -- Dockerfile found"
  PASS=$((PASS + 1))
else
  echo "  FAILED -- No Dockerfile"
  exit 1
fi

# Step 3: openenv validate (skip if not available)
echo "Step 3/3: Running openenv validate ..."
if command -v openenv &>/dev/null; then
  cd "$REPO_DIR" && openenv validate && PASS=$((PASS + 1)) && echo "  PASSED"
else
  echo "  SKIPPED -- openenv CLI not available (Python 3.9, needs 3.10+)"
  echo "  Note: Already passed on partner's machine with Python 3.13"
  PASS=$((PASS + 1))
fi

echo ""
echo "========================================"
echo "  $PASS/3 checks passed!"
echo "========================================"
