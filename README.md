# HelixNet: Protein Secondary Structure Prediction with CNN

HelixNet is a deep learning tool for **predicting protein secondary structure** (Helix, Sheet, Coil — the classic **Q3 classification**) directly from **amino acid sequences**.  

Using a **Convolutional Neural Network (CNN)** implemented in TensorFlow/Keras, HelixNet encodes protein sequences, performs a **70/30 train–test split**, and trains a robust model capable of recognizing structural motifs.  

This repository is **GitHub-ready**, easy to run, and fully reproducible.

---

## ✨ Features
- **Input**: Protein sequences in plain-text CSV format (`sequence,label`).
- **Encoding**: Converts amino acids (`ACDEFGHIKLMNPQRSTVWY`) into numerical tensors.
- **Architecture**: Embedding → Conv1D → Pooling → Dense → Softmax.
- **Training**: Stratified 70/30 split with early stopping.
- **Outputs**: 
  - Trained model (`artifacts/model.keras`)  
  - Metrics report (`artifacts/metrics.json`)  
  - Confusion matrix (`artifacts/confusion_matrix.npy`)  
- **CLI Tool**: Run with a single command (`protein-cnn`).

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/helixnet.git
cd helixnet

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

---

### 2. Prepare Your Data

Create a CSV file (e.g. `data/proteins.csv`) with the following format:

```csv
sequence,label
ACDEFGHIKLMNPQRSTVWY,C
HHHHHHHHHHHHHHHHHHHH,H
EEEEEEEEEEEEEEEEEEEE,E
```

- `sequence`: string of amino acids (A, C, D, …, Y)  
- `label`: Q3 secondary structure label:
  - **H** = α-Helix  
  - **E** = β-Sheet  
  - **C** = Coil  

A sample dataset is provided at `data/proteins.sample.csv`.

---

### 3. Train the Model

Run with the console script:
```bash
protein-cnn --csv data/proteins.csv --epochs 20 --batch_size 64 --artifacts artifacts
```

Or using the module:
```bash
python -m protein_cnn.cli --csv data/proteins.csv --epochs 20
```

---

### 4. Results & Outputs
After training, HelixNet saves results to the `artifacts/` folder:

- `model.keras` → trained TensorFlow model  
- `metrics.json` → accuracy, loss, classification report (per class)  
- `confusion_matrix.npy` → confusion matrix as NumPy array  

Example excerpt from `metrics.json`:
```json
{
  "test_accuracy": 0.8734,
  "test_loss": 0.4231,
  "classification_report": {
    "C": {"precision": 0.85, "recall": 0.88, "f1-score": 0.86},
    "E": {"precision": 0.87, "recall": 0.85, "f1-score": 0.86},
    "H": {"precision": 0.90, "recall": 0.89, "f1-score": 0.89}
  }
}
```

---

## ⚙️ Project Structure
```
helixnet/
├─ src/protein_cnn/
│  ├─ data.py         # Data encoding and sequence padding
│  ├─ model.py        # CNN model definition
│  ├─ train.py        # Training loop and evaluation
│  └─ cli.py          # Command-line interface
├─ data/
│  └─ proteins.sample.csv
├─ artifacts/         # Model + results (created after training)
├─ tests/             # Pytest smoke tests
├─ notebooks/         # Original research notebooks
├─ pyproject.toml
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## 🧪 Testing
Run the included tests:
```bash
pytest -q
```

Smoke test ensures:
- Training runs for at least 1 epoch on toy data.
- Model, metrics, and confusion matrix are saved correctly.

---

## 📊 Model Details
- **Embedding Layer**: Learns distributed representations of amino acids.  
- **Conv1D Layers**: Detect motifs across the sequence.  
- **Pooling**: Reduces dimensionality while retaining important features.  
- **Dense Layers**: Classifier mapping sequence motifs → Q3 labels.  
- **Output**: Softmax activation for 3-way classification (C/E/H).  

---

## 🌍 Use Cases
- Bioinformatics research  
- Structural biology coursework & teaching  
- Prototype for more advanced predictors (Q8, residue-level labeling, or hybrid CNN–RNN models)  

---

## 📌 Roadmap
- [ ] Add residue-level secondary structure prediction (Q3 per residue).  
- [ ] Extend to Q8 classification (DSSP categories).  
- [ ] Support transformer-based encoders (BERT-like embeddings).  
- [ ] Provide pretrained model weights for common datasets.  

---

## 📜 License
MIT License © 2025  
Feel free to use, modify, and distribute.
