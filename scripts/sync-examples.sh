#!/bin/bash
# Sync example files from the canonical irmf-examples repo into this repo.
#
# `irmf-examples` is authoritative. By default this copies only the TSL
# variants (the additive language), leaving every existing GLSL/WGSL file and
# the rust-only extras (sphere-cal, rodin-coil-2/3, ...) untouched.
#
# Usage:
#   scripts/sync-examples.sh [--tsl|--all] [--check]
#
#   --tsl    Copy only *-tsl.irmf files (default).
#   --all    Copy every .irmf file (never deletes extras).
#   --check  Report drift and exit 1 instead of copying.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES_DIR="${IRMF_EXAMPLES_DIR:-}"
if [ -z "$EXAMPLES_DIR" ]; then
  for candidate in \
      "$HOME/src/github.com/gmlewis/irmf-examples" \
      "$HOME/go/src/github.com/gmlewis/irmf-examples"; do
    if [ -d "$candidate/examples" ]; then
      EXAMPLES_DIR="$candidate"
      break
    fi
  done
fi
if [ -z "$EXAMPLES_DIR" ] || [ ! -d "$EXAMPLES_DIR/examples" ]; then
  echo "error: set IRMF_EXAMPLES_DIR to the irmf-examples checkout" >&2
  exit 1
fi

MODE="--tsl"
CHECK=0
for arg in "$@"; do
  case "$arg" in
    --tsl|--all) MODE="$arg" ;;
    --check) CHECK=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

SRC="$EXAMPLES_DIR/examples"
DST="$ROOT/examples"
drift=0
copied=0

copy_one() {
  local rel="$1"
  local src="$SRC/$rel"
  local dst="$DST/$rel"
  if [ ! -d "$(dirname "$dst")" ]; then
    return 0
  fi
  if [ -f "$dst" ] && cmp -s "$src" "$dst"; then
    return 0
  fi
  if [ "$CHECK" = "1" ]; then
    echo "drift: examples/$rel"
    drift=$((drift + 1))
  else
    cp "$src" "$dst"
    copied=$((copied + 1))
  fi
}

while IFS= read -r -d '' src; do
  rel="${src#"$SRC"/}"
  if [ "$MODE" = "--tsl" ] && [[ "$rel" != *-tsl.irmf ]]; then
    continue
  fi
  copy_one "$rel"
done < <(find "$SRC" \( -name '*.irmf' -o -name 'README.md' \) -print0)

if [ "$CHECK" = "1" ]; then
  if [ "$drift" -gt 0 ]; then
    echo "$drift file(s) out of sync; run scripts/sync-examples.sh $MODE" >&2
    exit 1
  fi
  echo "examples are in sync ($MODE)."
else
  echo "copied $copied file(s) from $SRC ($MODE)."
fi
