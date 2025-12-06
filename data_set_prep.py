import copy
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import tqdm
import utils

class HCQTDataset(Dataset):
    def __init__(self, config, debug=False, metadata_override=None):
        self.metadata = None
        self.sr = config["dataset"]["sample_rate"]
        self.data_path = Path(config["dataset"]["dataset_path"])
        self.bins_per_octave = config["dataset"]["bins_per_semitone"] * 12
        self.n_octaves = config["dataset"]["n_octaves"]
        self.n_time_frames = config["dataset"]["n_time_frames"]
        self.harmonics = config["dataset"].get("harmonics", [0.5, 1, 2, 3, 4])
        self.cqt_freq = librosa.cqt_frequencies(
            n_bins=self.bins_per_octave * self.n_octaves, 
            fmin=config["dataset"]["fmin"], 
            bins_per_octave=self.bins_per_octave
        )
        if metadata_override is not None:
            self.metadata = metadata_override
        else:
            json_path = config["dataset"].get("json_path")
            if not json_path:
                raise ValueError("json_path is required when no metadata_override is provided.")
            with open(json_path, "r") as f:
                self.metadata = json.load(f)

        self.samples = []
        self.labels = []
        for song_name, info in tqdm.tqdm(self.metadata.items(), total=len(self.metadata), desc="processing HCQTs and targets"):
            audio, _ = librosa.load((self.data_path / info["audio_path"]), sr=self.sr)
            pitch_df = pd.read_csv(self.data_path / info["pitch_path"], header=None, names=["seconds", "pitch"])
            
            hcqt, freqs_hz = utils.compute_hcqt(audio, self.sr,
                harmonics=self.harmonics,
                bins_per_octave=self.bins_per_octave,
                n_octaves = self.n_octaves,
                hop_length = config["dataset"]["hop_length"],
                fmin = config["dataset"]["fmin"]
            )
            times = librosa.frames_to_time(np.arange(hcqt.shape[1]), sr=self.sr, hop_length=config["dataset"]["hop_length"])
            target_salience = utils.f0_to_salience_time_major(pitch_df["seconds"], pitch_df["pitch"], freqs_hz, times, sigma_cents=25.0)

            for sample, label, _ in utils.iter_samples_for_track(hcqt, target_salience, self.n_time_frames):
                self.samples.append(sample)
                self.labels.append(label)

            if debug: break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X = torch.tensor(self.samples[idx], dtype=torch.float32)   # (C, n_time_frames, F)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)    # (n_time_frames, F, 1)
        return X, y

def get_dataloaders(config, debug=False):
    dataset = HCQTDataset(config, debug=debug)
    num_samples = len(dataset)
    split_idx = int(float(config["dataset"]["split"]) * num_samples)
    if num_samples > 1:
        split_idx = min(max(split_idx, 1), num_samples - 1)
    else:
        split_idx = 1

    indices = list(range(num_samples))
    train_subset = torch.utils.data.Subset(dataset, indices[:split_idx])
    val_subset = torch.utils.data.Subset(dataset, indices[split_idx:])

    train_loader = DataLoader(train_subset, batch_size=config["model"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config["model"]["batch_size"], shuffle=False)

    return train_loader, val_loader


def build_audio_pitch_metadata(dataset_root, audio_subdir="Audio", pitch_subdir="Annotations/F0", pitch_suffix="_f0.csv"):
    dataset_root = Path(dataset_root)
    audio_dir = dataset_root / audio_subdir
    pitch_dir = dataset_root / pitch_subdir

    metadata = {}
    for audio_file in sorted(audio_dir.glob("*.wav")):
        track_stem = audio_file.stem
        pitch_file = pitch_dir / f"{track_stem}{pitch_suffix}"

        metadata[track_stem] = {
            "audio_path": audio_file.relative_to(dataset_root).as_posix(),
            "pitch_path": pitch_file.relative_to(dataset_root).as_posix(),
        }

    return metadata

def get_evaluation_dataloader(config, dataset_root, batch_size=None, debug=False):
    eval_config = copy.deepcopy(config)
    eval_config["dataset"]["dataset_path"] = str(dataset_root)

    metadata = build_audio_pitch_metadata(dataset_root)
    dataset = HCQTDataset(eval_config, debug=debug, metadata_override=metadata)

    effective_batch_size = batch_size or config["model"]["batch_size"]
    return DataLoader(dataset, batch_size=effective_batch_size, shuffle=False)

if __name__ == "__main__":
    import yaml
    with open("configs/cqt.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    dataset = HCQTDataset(config, debug=True)
    X, y = dataset.__getitem__(0)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    for sample, label, t0 in utils.iter_samples_for_track(X, y, 50):
        print("Plotting window starting at:", t0)
        harmonic_labels = [f"{h}×" for h in dataset.harmonics]
        utils.plot_sample_and_label(sample, label, harmonic_labels=harmonic_labels)
        break