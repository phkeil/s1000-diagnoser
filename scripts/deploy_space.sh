#!/usr/bin/env bash
# Usage: ./scripts/deploy_space.sh
#
# Stages everything the Hugging Face Space needs but does not track in this
# repo: the exported ONNX model, its thresholds, and the Python sources the
# Pyodide worker executes.
#
# The Python sources are COPIED rather than imported across a directory
# boundary because the deployed Space is rooted at space/ - it has no parent to
# reach into. They are copies of the real files, refreshed on every deploy, and
# gitignored here so nobody edits the copy by mistake: src/data.py is the
# original, space/python/src/data.py is a build artifact.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPACE_DIR="$REPO_ROOT/space"

cd "$REPO_ROOT"

echo "==> Exporting ONNX model"
python -m src.export_onnx

echo "==> Staging model artifacts into space/assets/"
mkdir -p "$SPACE_DIR/assets"
cp "$REPO_ROOT/models/s1000_cae.onnx" "$SPACE_DIR/assets/s1000_cae.onnx"

# Optional: a checkpoint trained before src/thresholds.py existed has none, and
# the app falls back to config.yaml's anomaly values exactly as the server does.
if [ -f "$REPO_ROOT/models/thresholds.json" ]; then
  cp "$REPO_ROOT/models/thresholds.json" "$SPACE_DIR/assets/thresholds.json"
  echo "    thresholds.json staged"
else
  rm -f "$SPACE_DIR/assets/thresholds.json"
  echo "    WARNING: models/thresholds.json not found - the Space will fall back to"
  echo "             config/config.yaml's anomaly values. Run 'python -m src.train"
  echo "             --write-local-checkpoint' to produce model-specific thresholds."
fi

echo "==> Staging Python preprocessing sources into space/python/"
mkdir -p "$SPACE_DIR/python/src" "$SPACE_DIR/python/config"
for module in __init__.py config.py manifest.py audio_utils.py data.py; do
  cp "$REPO_ROOT/src/$module" "$SPACE_DIR/python/src/$module"
done
cp "$REPO_ROOT/config/config.yaml" "$SPACE_DIR/python/config/config.yaml"

MODEL_SIZE="$(du -h "$SPACE_DIR/assets/s1000_cae.onnx" | cut -f1)"
echo
echo "Staged. Model: $MODEL_SIZE"
echo
echo "Test locally:"
echo "  cd space && python -m http.server 8080   # then open http://localhost:8080/"
echo
echo "Push to Hugging Face (space/ is its own git repo - see space/README.md):"
echo "  cd space && git add -A && git commit -m 'Deploy' && git push huggingface main"
