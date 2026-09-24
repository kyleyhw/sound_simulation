#!/bin/bash
# SessionStart hook for Claude Code on the web: installs the Python env
# (uv, dev + ml extras, CPU torch) and the frontend npm packages so the
# kernel gates, learning tests, ruff/ty and tsc work from the first turn.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Python: .venv/ from uv.lock. Idempotent — a no-op when already synced.
uv sync --extra dev --extra ml

# Frontend: npm ci, not install. `npm install` rewrites package-lock.json
# (npm-version metadata churn), which would dirty the tree every session.
# ~6 s for this small dependency set.
(cd frontend && npm ci --no-audit --no-fund)

# Make `python`, `ruff`, `ty` resolve to the project venv for the session.
echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
