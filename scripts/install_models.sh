#!/usr/bin/env bash
# Install spaCy language models via direct GitHub release URLs.
# Using direct pip URLs instead of `python -m spacy download` because
# spaCy's internal compatibility table lookup silently returns empty versions
# for some patch releases (e.g. 3.7.5), producing broken 404 URLs.
set -euo pipefail

SPACY_VERSION=$(python -c "import spacy; print(spacy.__version__)")
echo ">>> Detected spaCy $SPACY_VERSION"

SM_URL="https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
TRF_URL="https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3-py3-none-any.whl"

echo ">>> Installing en_core_web_sm (small, CPU-fast)..."
pip install "$SM_URL"

echo ">>> Installing en_core_web_trf (transformer, GPU-accelerated)..."
pip install "$TRF_URL"

echo ">>> Verifying installs..."
python - <<'EOF'
import spacy
for model in ("en_core_web_trf", "en_core_web_sm"):
    nlp = spacy.load(model)
    print(f"  OK  {model}  v{nlp.meta['version']}")
EOF

echo ">>> spaCy models ready."
