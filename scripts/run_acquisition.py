#!/usr/bin/env python3
"""Run OpenBCI acquisition with visual paradigm."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import scripts._bootstrap  # noqa: F401

from src.acquisition.brainflow_stream import BrainFlowStream
from src.acquisition.data_writer import save_eeg_csv, save_events_csv
from src.acquisition.openbci_config import OpenBCIConfig
from src.acquisition.paradigm_gui import ParadigmGUI
from src.acquisition.trial_scheduler import generate_session_schedule
from src.utils.config import load_config, project_root
from src.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/acquisition.yaml")
    parser.add_argument("--mock", action="store_true", help="No hardware; mock EEG stream")
    parser.add_argument("--no-gui", action="store_true", help="Skip pygame (timing only)")
    args = parser.parse_args()

    logger = setup_logger()
    root = project_root()
    cfg = load_config(root / args.config)
    obci = OpenBCIConfig.from_dict(cfg)
    out_sess = root / obci.output_dir / args.subject / f"session_{args.session:02d}"
    out_sess.mkdir(parents=True, exist_ok=True)

    schedule = generate_session_schedule(
        args.subject, args.session,
        n_trials=cfg.get("trials_per_session", 50),
    )

    stream = BrainFlowStream(obci, mock=args.mock)
    stream.prepare_session()
    stream.start_stream()

    gui = None if args.no_gui else ParadigmGUI()
    eeg_chunks = []
    ts_chunks = []
    event_rows = []
    global_t = 0.0

    try:
        for _, row in schedule.iterrows():
            cue = row["cue"]
            if gui:
                times = gui.run_trial(cue)
            else:
                time.sleep(10.0)
                times = {"trial_start": time.perf_counter()}

            duration = 10.0
            data = stream.stream_for_duration(duration, poll_hz=50)
            n_samp = data.shape[1]
            ts = global_t + np.arange(n_samp) / obci.sampling_rate
            eeg_chunks.append(data.T)
            ts_chunks.append(ts)
            global_t = float(ts[-1]) + 1.0

            event_rows.append({
                **row.to_dict(),
                "trial_start": global_t - duration - 1.0,
                "cue_start": row["cue_start"] + (global_t - duration - 10),
            })
    finally:
        stream.stop_stream()
        stream.release()
        if gui:
            gui.close()

    eeg_all = np.vstack(eeg_chunks)
    ts_all = np.concatenate(ts_chunks)
    ch_names = obci.channel_names[: eeg_all.shape[1]]
    save_eeg_csv(out_sess / "eeg_raw.csv", ts_all, eeg_all, ch_names)
    save_events_csv(out_sess / "events.csv", schedule)
    logger.info("Acquisition saved to %s", out_sess)


if __name__ == "__main__":
    main()
