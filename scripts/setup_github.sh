#!/usr/bin/env bash
# Vincula este proyecto con GitHub y prepara el primer push.
# Uso: ./scripts/setup_github.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REMOTE="git@github.com:manuelquistial/modelo_bilstm.git"

if [[ ! -d .git ]]; then
  git init -b main
  echo "Repositorio git inicializado (rama main)."
fi

if git remote get-url origin &>/dev/null; then
  git remote set-url origin "$REMOTE"
else
  git remote add origin "$REMOTE"
fi

echo "Remote origin: $(git remote get-url origin)"

git add -A
git status --short | head -40

if git diff --cached --quiet; then
  echo "No hay cambios para commitear."
else
  git commit -m "$(cat <<'EOF'
Initial commit: Sun et al. 2026 lower-limb MI BCI pipeline.

Includes CNN-SE-BiLSTM, baselines, PSO, preprocessing, Paperspace scripts,
and paper replication documentation.
EOF
)"
  echo "Commit creado."
fi

echo ""
echo "Siguiente paso (en tu máquina, con SSH configurado en GitHub):"
echo "  git push -u origin main"
