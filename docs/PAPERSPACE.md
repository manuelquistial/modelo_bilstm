# Ejecución en Paperspace

Repositorio: `git@github.com:manuelquistial/modelo_bilstm.git`

## 1. Crear máquina

Recomendado:

- **Gradient Notebook** o **GPU Machine** (Ubuntu 22.04)
- GPU: **A4000 / RTX4000 / A5000** o superior (16 GB+ VRAM)
- Disco: **≥ 20 GB**
- Plantilla: **PyTorch** (Python 3.10+)

## 2. Clonar el repositorio

```bash
cd /notebooks   # o ~
git clone git@github.com:manuelquistial/modelo_bilstm.git
cd modelo_bilstm
```

Si usas HTTPS:

```bash
git clone https://github.com/manuelquistial/modelo_bilstm.git
cd modelo_bilstm
```

## 3. Entorno Python (importante: orden de instalación)

En Paperspace **no** instales `torch` con `pip install -r requirements.txt` antes del fix: mezcla wheels de NVIDIA y rompe `libnvJitLink`.

```bash
cd /notebooks/modelo_bilstm
git pull

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

# 1) Dependencias SIN torch
pip install -r requirements-paperspace.txt

# 2) PyTorch GPU alineado (prueba cu124 → cu121 → cu118)
chmod +x scripts/fix_paperspace_torch.sh
./scripts/fix_paperspace_torch.sh
```

Verifica:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Si sigue `undefined symbol: __nvJitLink...`

**Opción A — Reinstalar en limpio:**

```bash
source .venv/bin/activate
./scripts/fix_paperspace_torch.sh
```

**Opción B — Venv con PyTorch del sistema (plantilla Gradient PyTorch):**

```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements-paperspace.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Opción C — Solo CPU (entrena lento, sin GPU):**

```bash
TORCH_CPU=1 ./scripts/fix_paperspace_torch.sh
```

El import BNCI **no necesita** PyTorch; el entrenamiento sí.

## 4. Pipeline completo (datos sintéticos)

```bash
chmod +x scripts/paperspace_run_all.sh
./scripts/paperspace_run_all.sh
```

O paso a paso:

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Default: BNCI2014_001 (MOABB; primera vez descarga ~1 GB)
python scripts/import_bnci2014_001.py --output data/processed

# Alternativa sin descarga:
# DATA_SOURCE=synthetic ./scripts/paperspace_run_all.sh
python scripts/run_training_all_subjects.py --model proposed
python scripts/run_baselines.py --all-subjects --quick-epochs 50
python scripts/generate_report.py
pytest tests/ -q
```

## 5. Entrenamiento completo (paper: 400 épocas)

```bash
python scripts/run_training_all_subjects.py --model proposed
```

Tiempo estimado: varias horas en GPU para 10 sujetos.

## 6. PSO (opcional)

```bash
python scripts/run_pso.py --subject S01
```

## 7. Persistencia de resultados

Los artefactos quedan en:

- `data/processed/` — trials preprocesados
- `results/metrics/` — CSV de métricas
- `results/models/` — checkpoints `.pt`
- `results/reports/` — informes y figuras

En Paperspace, copia a **Persistent Storage** o descarga:

```bash
tar -czf results_bundle.tar.gz results/
```

## 8. Variables de entorno útiles

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export OMP_NUM_THREADS=4
```

## 9. OpenBCI en Paperspace

La adquisición en vivo requiere hardware local; en la nube usa solo el modo **sintético** o sube `data/raw/` desde tu PC:

```bash
# En tu máquina local
scp -r data/raw/S01 paperspace@<host>:~/modelo_bilstm/data/raw/
```

## 10. Problemas frecuentes

| Problema | Solución |
|----------|----------|
| `No module named 'src'` | `export PYTHONPATH=$PWD` |
| CUDA OOM | Reduce `batch_size` en `configs/training.yaml` |
| pygame sin display | Usa `--no-gui` en adquisición o solo pipeline sintético |
| MNE lento en CPU | Usa máquina con GPU; ICA es CPU-bound |
| `libnvJitLink.so.12: undefined symbol` | `pip install -r requirements-paperspace.txt` luego `./scripts/fix_paperspace_torch.sh` (no uses `requirements.txt` antes del fix) |
| `import_bnci2014_001` falla con error de `torch` | Actualiza el repo (`git pull`); el import no debe cargar CUDA |
