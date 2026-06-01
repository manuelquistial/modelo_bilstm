# Sun et al. 2026 — Lower-Limb Motor Imagery BCI

Reproducible implementation of **“Bridging minds and limbs: novel hybrid deep learning approach for low-cost EEG-based lower limb motor imagery classification”** (Sun et al., 2026).

**Repositorio:** [github.com/manuelquistial/modelo_bilstm](https://github.com/manuelquistial/modelo_bilstm)  
**Clone SSH:** `git@github.com:manuelquistial/modelo_bilstm.git`

Binary task: **class 0** = imagine lifting **left** thigh; **class 1** = imagine lifting **right** thigh.

## Paperspace (GPU en la nube)

Guía completa: [docs/PAPERSPACE.md](docs/PAPERSPACE.md)

```bash
git clone git@github.com:manuelquistial/modelo_bilstm.git
cd modelo_bilstm
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
export PYTHONPATH="${PWD}:${PYTHONPATH}"
./scripts/paperspace_run_all.sh
```

Para entrenamiento completo del paper (400 épocas): `python scripts/run_training_all_subjects.py --model proposed`

## What this project implements

| Component | Description |
|-----------|-------------|
| Acquisition | OpenBCI Cyton via BrainFlow + pygame paradigm |
| Preprocessing | 1000→125 Hz, 1–40 Hz, ICA (MNE), MI epoch 2–6 s → `(501, 15)` |
| Augmentation | Overlapping windows: 251 samples, step 50 → 6 segments/trial |
| Split | Intra-subject stratified 70/30 **before** segmentation |
| Model | CNN-SE-BiLSTM per **Table 1**: CNN→SE→Flatten **13080**→BiLSTM(16)→FC+tanh→2 |
| Optimization | PSO on conv filters/kernels (12–16, 10×10–15×15) |
| Baselines | CSP+SVM [38], Eff-KNN [39], EEGNet [36], ShallowConvNet [37], CNN-LSTM |
| Metrics | Accuracy, Cohen κ, sensitivity, confusion matrices (segment & trial) |
| Analysis | PSD topomaps (μ, β), ERD/ERS at C3/C4 |

## Hardware (real acquisition)

- OpenBCI Cyton (16 ch), WiFi, 1000 Hz
- 15 EEG channels (10–20): Fz, F3, F4, F7, F8, Cz, C3, C4, T3, T4, Pz, P3, P4, T5, T6
- Reference/ground on earlobes (not used as input)

## Clonar desde GitHub

```bash
git clone git@github.com:manuelquistial/modelo_bilstm.git
cd modelo_bilstm
```

Primera vez (vincular carpeta local existente al remoto):

```bash
chmod +x scripts/setup_github.sh
./scripts/setup_github.sh
git push -u origin main
```

## Installation

```bash
cd modelo_bilstm   # project root
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start (synthetic data — no hardware)

```bash
# 1. Synthetic raw EEG (10 subjects × 3 sessions × 50 trials)
python scripts/generate_synthetic_data.py

# 2. Preprocess → trials.npz (150, 501, 15) per subject
python scripts/run_preprocessing.py --input data/sample --output data/processed

# 3. Train proposed model (one subject; use --quick-epochs 5 for smoke test)
python scripts/run_training_subject.py --subject S01 --model proposed --quick-epochs 5

# 4. All baselines (uses quick epochs for DL by default)
python scripts/run_baselines.py --all-subjects

# 5. PSO (optional; slow)
python scripts/run_pso.py --subject S01

# 6. PSD / ERD analysis
python scripts/run_psd_analysis.py --subject S01
python scripts/run_erd_analysis.py --subject S01

# 7. Report
python scripts/generate_report.py

# Tests
pytest tests/ -q
```

## Real OpenBCI acquisition

```bash
python scripts/run_acquisition.py --subject S01 --session 1 --config configs/acquisition.yaml
# Mock (no hardware):
python scripts/run_acquisition.py --subject S01 --session 1 --mock --no-gui
```

Edit `configs/acquisition.yaml` for `board_id`, `serial_port`, or WiFi `ip_address` / `ip_port`.

## Training (full paper settings)

```bash
python scripts/run_training_all_subjects.py --model proposed
# 400 epochs, Adam lr 0.01 → 0.001 at epoch 300 (see configs/training.yaml)
```

## Data layout

```
data/raw/S01/session_01/
  eeg_raw.csv      # timestamp + 15 channels @ 1000 Hz
  events.csv       # trial timing and labels
data/processed/S01/
  trials.npz       # X (150,501,15), y, trial_ids, session_ids
```

## Model input shapes (Table 1)

| Stage | Shape |
|-------|--------|
| Trial (processed) | `(150, 501, 15)` |
| Train segment (expansion) | `(630, 251, 15)` per subject (105×6) |
| Test trials | `(45, 501, 15)` — metrics at **trial level** |
| CNN input | `(batch, 251, 15)` |
| After Flatten | `(batch, 13080)` |
| BiLSTM | `(batch, 1, 13080)` → hidden 16×2 |
| Logits | `(batch, 2)` |

**Paper protocol:** overlapping windows **only on training** (§2.3); `channel_normalize: false` by default.

**Architecture:** Conv 16×(10,10) → MaxPool (10,1) → Conv 12×(15,15) → SE (r=3) → Flatten 13080 → BiLSTM → FC+tanh → FC 2.

## Evaluation modes

1. **Segment-level:** metrics on all sliding windows.
2. **Trial-level:** mean softmax over 6 segments per trial, then argmax.

Normalization: channel z-score fit **only** on training segments.

## Paper reference metrics (not hardcoded)

| Model | Accuracy | Kappa | Sensitivity |
|-------|----------|-------|-------------|
| CSP+SVM | 0.538 | 0.095 | 0.549 |
| EEGNet | 0.637 | 0.271 | 0.614 |
| ConvNet | 0.650 | 0.305 | 0.623 |
| KNN | 0.596 | 0.206 | 0.569 |
| CNN-LSTM | 0.680 | 0.348 | 0.675 |
| **Proposed** | **0.721** | **0.436** | **0.699** |

Exact numbers require the authors’ private dataset; synthetic data will not match these values.

## PSO

Optimizes: `conv1/conv2` filter counts (12–16), kernel sizes (10–15).  
Coefficients: `w=0.729`, `c1=c2=1.49445`.  
Output: `results/pso/best_params_subject_{id}.json`

## Limitations

- Small cohort (10 subjects), intra-subject only — no cross-subject generalization in the paper protocol.
- No clinical validation.
- ICA may be skipped on failure (logged warning).
- **Never** apply sliding windows before train/test split (data leakage).
- Do not use test data for normalization or PSO fitness on test.

## License

Research / educational use. Cite Sun et al. (2026) if you use this code.
