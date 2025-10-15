#!/usr/bin/env bash
set -euo pipefail

# Render all Mermaid (.mmd) diagrams in docs/trees/ to SVG and PNG
# Requires: mermaid-cli (mmdc)
# Install:  npm i -g @mermaid-js/mermaid-cli
#
# Usage:
#   chmod +x scripts/render_trees.sh
#   ./scripts/render_trees.sh

# Resolve paths relative to this script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TREES_DIR="$REPO_ROOT/docs/trees"
EXPORT_DIR="$TREES_DIR/exports"

if ! command -v mmdc >/dev/null 2>&1; then
  echo "Error: 'mmdc' (Mermaid CLI) not found."
  echo "Install it with: npm i -g @mermaid-js/mermaid-cli"
  exit 1
fi

mkdir -p "$EXPORT_DIR"

shopt -s nullglob
MMDS=( "$TREES_DIR"/*.mmd )
if [ ${#MMDS[@]} -eq 0 ]; then
  echo "No .mmd files found in: $TREES_DIR"
  exit 0
fi

echo "Rendering Mermaid diagrams to: $EXPORT_DIR"
for mmd in "${MMDS[@]}"; do
  base="$(basename "$mmd" .mmd)"
  svg_out="$EXPORT_DIR/$base.svg"
  png_out="$EXPORT_DIR/$base.png"

  echo "  - $base.mmd -> $base.svg"
  mmdc -i "$mmd" -o "$svg_out" --backgroundColor transparent

  echo "  - $base.mmd -> $base.png"
  mmdc -i "$mmd" -o "$png_out" --backgroundColor transparent
done

echo "Done."
