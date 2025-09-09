
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import List, Tuple

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 is padding
LABELS = ["C", "E", "H"]  # coil, sheet, helix
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}

def encode_sequence(seq: str) -> np.ndarray:
    """Map amino-acid characters to integer ids; unknowns become 0 (pad)."""
    return np.array([AA_TO_ID.get(ch, 0) for ch in seq], dtype=np.int32)

def load_csv(csv_path: str) -> Tuple[list[np.ndarray], np.ndarray]:
    """Load a CSV with columns: sequence,label (labels ∈ {C,E,H})."""
    df = pd.read_csv(csv_path)
    if not {"sequence", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'sequence' and 'label' columns.")
    df = df.dropna(subset=["sequence", "label"]).copy()
    df["sequence"] = (
        df["sequence"].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    )
    # Filter non-Q3 labels
    df = df[df["label"].isin(LABELS)]
    X = [encode_sequence(s) for s in df["sequence"].tolist()]
    y = df["label"].map(LABEL_TO_ID).values
    return X, y

def pad_sequences(int_sequences: List[np.ndarray], max_len: int | None = None):
    """Pad integer-encoded sequences to a uniform length with zeros."""
    if len(int_sequences) == 0:
        raise ValueError("No sequences provided.")
    if max_len is None:
        max_len = max(len(s) for s in int_sequences)
    X = np.zeros((len(int_sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(int_sequences):
        trunc = seq[:max_len]
        X[i, :len(trunc)] = trunc
    return X, max_len
