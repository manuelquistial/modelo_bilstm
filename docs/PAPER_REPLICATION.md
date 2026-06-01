# Paper replication checklist — Sun et al. 2026

This document maps each methodological item from the paper to the codebase.

## Implemented as specified

| Paper section | Requirement | Code |
|---------------|-------------|------|
| §2.1 | OpenBCI Cyton 16 ch, 1000 Hz, WiFi | `acquisition/` |
| §2.2 | Paradigm 2+4+4 s, 150 trials, paired blocks | `paradigm_gui.py`, `trial_scheduler.py` |
| §2.3 | Downsample 125 Hz, 1–40 Hz, ICA ART-like | `preprocessing_pipeline.py`, `artifact_removal.py` |
| §2.3 | Epoch 2–6 s → 501×15 | `epoching.py` |
| §2.3 | Split 70/30 intra-subject | `subject_split.py` |
| §2.3 | **Data expansion only during training** | `dataset.yaml` `paper_mode: true`, `data_loader.py` |
| §2.4 | Windows 251, step 50, 6 segments | `augmentation.py` |
| Table 1 | CNN 16×(10,10), pool (10,1), CNN 12×(15,15), SE r=3 | `paper_cnn_frontend.py` |
| Table 1 | Flatten 13080 → BiLSTM 16 → FC tanh → 2 | `cnn_se_bilstm.py` |
| §2.5.5 D | Adam 0.01, drop to 0.001 at epoch 300, 400 epochs | `training.yaml`, `trainer.py` |
| §2.5.3 | PSO kernels 12–16, size 10–15 | `optimization/pso.py` |
| §2.6 | EEGNet [36], ConvNet [37], CSP-SVM [38], Eff-KNN [39], CNN-LSTM | `models/`, `baselines/` |
| §2.6 | Grid search for classical ML | `csp_svm.py` |
| Eq. 10–13 | Accuracy, κ, sensitivity | `evaluation/metrics.py` |
| §3.4–3.5 | PSD μ/β, ERD C3/C4 | `analysis/` |

## Known MATLAB ↔ PyTorch differences

1. **Flatten 13080**: Table 1 fixes this size. If native conv output ≠ 13080, a linear adapter `proj` maps to 13080 (logged at init). Re-run with authors' MATLAB export to verify exact padding.
2. **ICA**: EEGLAB ART vs MNE ICA + frontal/kurtosis rules — functionally equivalent, not byte-identical.
3. **No z-score** in paper pipeline — disabled via `channel_normalize: false`.

## Primary metric for comparison with Table 2/3

Use **trial-level** accuracy / κ / `sensitivity_paper` on **45 test trials** per subject.
