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
    def __init__(self, config, debug=False):
        self.metadata = None
        self.sr = config["dataset"]["sample_rate"]
        self.data_path = Path(config["dataset"]["dataset_path"])
        self.bins_per_octave = config["dataset"]["bins_per_semitone"] * 12
        self.n_octaves = config["dataset"]["n_octaves"]
        self.cqt_freq = librosa.cqt_frequencies(
            n_bins=self.bins_per_octave * self.n_octaves, 
            fmin=config["dataset"]["fmin"], 
            bins_per_octave=self.bins_per_octave
        )
        with open(config["dataset"]["json_path"], "r") as f:
            self.metadata = json.load(f)

        self.songs = []
        self.hcqts = []
        self.targets = []
        for song_name, info in tqdm.tqdm(self.metadata.items(), total=len(self.metadata), desc="processing HCQTs and targets"):
            self.songs.append(song_name)
            audio, _ = librosa.load((self.data_path / info["audio_path"]), sr=self.sr)
            pitch_df = pd.read_csv(self.data_path / info["pitch_path"], header=None, names=["seconds", "pitch"])
            
            hcqt, freqs_hz = utils.compute_hcqt(audio, self.sr,
                bins_per_octave=self.bins_per_octave,
                n_octaves = self.n_octaves,
                hop_length = config["dataset"]["hop_length"],
                fmin = config["dataset"]["fmin"]
            )
            times = librosa.frames_to_time(np.arange(hcqt.shape[1]), sr=self.sr, hop_length=config["dataset"]["hop_length"])
            target_salience = utils.f0_to_salience_time_major(pitch_df["seconds"], pitch_df["pitch"], freqs_hz, times, sigma_cents=25.0)

            self.hcqts.append(hcqt)
            self.targets.append(target_salience)
            if debug:
                break

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        X = torch.tensor(self.hcqts[idx], dtype=torch.float32)   # (C, T, F)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)    # (T, F, 1)
        # metadata = self.metadata[str(self.songs[idx])]
        # return X, y, metadata
        return X, y

def get_dataloaders(config):
    dataset = HCQTDataset(config)
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
        utils.plot_sample_and_label(sample, label)
        break