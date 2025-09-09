
import argparse
from .train import train

def main():
    parser = argparse.ArgumentParser(description="Protein Q3 CNN trainer")
    parser.add_argument("--csv", type=str, default="data/proteins.csv",
                        help="Path to CSV with columns: sequence,label (labels in {C,E,H})")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--artifacts", type=str, default="artifacts")
    args = parser.parse_args()

    metrics = train(csv_path=args.csv, epochs=args.epochs, batch_size=args.batch_size, artifacts_dir=args.artifacts)
    print(f"\nTest accuracy: {metrics['test_accuracy']:.4f} | Test loss: {metrics['test_loss']:.4f}")
    print("Saved model and metrics to:", args.artifacts)
