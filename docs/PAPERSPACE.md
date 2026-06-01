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

## 3. Entorno Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

# PyTorch con CUDA (ajusta cu121/cu118 según la imagen de Paperspace)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Verifica GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

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
