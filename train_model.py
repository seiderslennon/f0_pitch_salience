import argparse
from pathlib import Path
import torch
from torch import nn
import data_set_prep
import utils
import yaml
import matplotlib.pyplot as plt

class PitchSalience(nn.Module):
    def __init__(self):
        super(PitchSalience, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, (3, 3), padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, (3, 3), padding="same")
        self.relu = nn.ReLU()

    def forward(self, x):
        # input is (batch, channels, time, freq)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)  # (batch, 1, time, freq)

        x = torch.transpose(x, 1, 2)  # (batch, time, 1, freq)
        x = torch.transpose(x, 2, 3)  # (batch, time, freq, 1)
        # no sigmoid at the end here, because we are using BCEWithLogitsLoss
        return x

def train_model(config, model, train_loader, val_loader, device):
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.ones([1], device=device) * 10)
    optimizer = torch.optim.Adam(model.parameters(), config["model"]["learning_rate"])

    for epoch in range(config["model"]["num_epochs"]):
        print(f"Epoch {epoch+1}")
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(device)                # (B, C, T, F)
            y = y.to(device)                # (B, T, F, 1)

            pred = model(X)                 # (B, T, F, 1)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}, loss = {loss.item():.4f}")

        # simple validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
        val_loss /= max(1, len(val_loader))
        print(f"  Validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pitch salience model.")
    parser.add_argument(
        "--config",
        default="configs/cqt.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--debug-dataset",
        action="store_true",
        help="Load only a single track to speed up iteration.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    exp_name = str(config["experiment_name"])
    base_model_path = Path(exp_name)
    if base_model_path.suffix != ".pth":
        base_model_path = base_model_path.with_suffix(".pth")

    model_path = base_model_path
    if model_path.exists():
        counter = 0
        while True:
            # ensure each training run gets a unique checkpoint filename
            candidate = base_model_path.with_name(f"{base_model_path.stem}_{counter}{base_model_path.suffix}")
            if not candidate.exists():
                model_path = candidate
                break
            counter += 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = PitchSalience().to(device)
    train_loader, val_loader = data_set_prep.get_dataloaders(
        config, debug=args.debug_dataset
    )

    train_model(config, model, train_loader, val_loader, device)
    torch.save(model, model_path)

    X_batch, y_batch = next(iter(train_loader))
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    model.eval()
    with torch.no_grad():
        predicted_salience = model(X_batch)
    utils.visualize(predicted_salience, X_batch, y_batch)
