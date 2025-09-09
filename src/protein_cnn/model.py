
from __future__ import annotations
from tensorflow.keras import layers, models

def build_cnn(vocab_size: int, max_len: int, num_classes: int = 3) -> models.Model:
    """Conv1D classifier for Q3 (sequence-level)."""
    model = models.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(input_dim=vocab_size + 1, output_dim=64, mask_zero=True),
        layers.Conv1D(128, kernel_size=7, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
