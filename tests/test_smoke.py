
import os, json, numpy as np
from protein_cnn.train import train

def test_train_smoke(tmp_path):
    # Create a tiny CSV
    csv = tmp_path / "mini.csv"
    csv.write_text("sequence,label\nHHHHHHHHHH,H\nEEEEEEEEEE,E\nCCCCCCCCCC,C\n")
    artifacts = tmp_path / "artifacts"
    metrics = train(str(csv), epochs=1, batch_size=4, artifacts_dir=str(artifacts))
    assert "test_accuracy" in metrics
    assert os.path.exists(artifacts / "model.keras")
    assert os.path.exists(artifacts / "metrics.json")
