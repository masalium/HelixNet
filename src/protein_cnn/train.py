
from __future__ import annotations
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from .data import AMINO_ACIDS, LABELS, load_csv, pad_sequences
from .model import build_cnn

def train(csv_path: str, epochs: int = 20, batch_size: int = 64, artifacts_dir: str = "artifacts") -> dict:
    os.makedirs(artifacts_dir, exist_ok=True)

    X_list, y = load_csv(csv_path)
    X_padded, max_len = pad_sequences(X_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=0.30, random_state=42, stratify=y
    )

    model = build_cnn(vocab_size=len(AMINO_ACIDS), max_len=max_len, num_classes=len(LABELS))
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")]

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    report = classification_report(y_test, y_pred, target_names=LABELS, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(LABELS))))

    # Save model and history
    model.save(os.path.join(artifacts_dir, "model.keras"))
    np.save(os.path.join(artifacts_dir, "confusion_matrix.npy"), cm)
    # Save metrics json
    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "classification_report": report,
        "max_len": int(max_len),
    }
    import json
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
