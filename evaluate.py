import argparse
import torch
from torch import nn
import yaml
import data_set_prep
from models import PitchSalience, PitchSalienceUnet
import utils


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained pitch salience model.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/cqt.yaml")
    parser.add_argument("--eval_root", type=str, default="Data/Evaluation/vocadito")
    parser.add_argument("--log-file", type=str, default="evaluation_output.txt")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    map_location = None if device == "cuda" else torch.device("cpu")
    model = torch.load(args.model_path, map_location=map_location, weights_only=False)
    model = model.to(device)
    model.eval()

    eval_loader = data_set_prep.get_evaluation_dataloader(
        config,
        dataset_root=args.eval_root,
        batch_size=args.batch_size,
        debug=args.debug,
    )

    X_batch, y_batch = next(iter(eval_loader))
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        predicted_salience_logits = model(X_batch)
        predicted_salience = torch.sigmoid(predicted_salience_logits)

    utils.visualize(predicted_salience_logits, X_batch, y_batch, "evaluation_sample.png")
    reshaped = (torch.squeeze(predicted_salience, [0, 3])).numpy()
    print(reshaped.shape)
    print(reshaped)


if __name__ == "__main__":
    main()
