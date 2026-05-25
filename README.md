# 🧬 AF Gene Sequence Classifier

A machine learning pipeline that classifies protein sequences into **16 Atrial Fibrillation (AF)-associated genes** using ProtBERT-BFD embeddings combined with physicochemical features. Deployed as an interactive Streamlit web application.

---

## 🫀 Background

Atrial fibrillation is the most common sustained cardiac arrhythmia, affecting over 37 million people worldwide and raising stroke risk 5-fold. This project automates gene classification from raw protein sequence data — useful for annotating novel or synthetic variants — by combining transformer-based protein language model embeddings with handcrafted biophysical features.

---

## 🎯 Target Genes (16)

| Gene | Protein | Role in AF |
|------|---------|------------|
| SCN5A | Nav1.5 | Primary cardiac sodium channel |
| KCNQ1 | Kv7.1 | Slow delayed rectifier K⁺ |
| RYR2 | Ryanodine Receptor 2 | SR Ca²⁺ release |
| KCNH2 | hERG | Rapid delayed rectifier K⁺ |
| KCNA5 | Kv1.5 | Ultra-rapid K⁺ (IKur) |
| KCNJ2 | Kir2.1 | Inward rectifier K⁺ |
| NPPA | ANP | Atrial natriuretic peptide |
| GJA5 | Connexin 40 | Atrial gap junction |
| PITX2 | PITX2 | Homeobox TF — left atrial development |
| PRKAG2 | AMPKγ | AMP kinase — Wolff-Parkinson-White |
| MYL4 | MLC4 | Atrial myosin light chain |
| TBX5 | TBX5 | T-box TF — Holt-Oram syndrome |
| SCN1B | Nav β1 | Sodium channel beta subunit |
| SCN2B | Nav β2 | Sodium channel beta subunit |
| SCN3B | Nav β3 | Sodium channel beta subunit |
| SCN4B | Nav β4 | Sodium channel beta subunit |

---

## 🏗️ Pipeline Overview

```
UniProt Swiss-Prot CSVs
        │
        ├─► Data Cleaning & Validation
        │       └─► Gene mapping, sequence validation (20–5000 AA)
        │
        ├─► Preprocessing Benchmark
        │       ├─► k-mer Cosine Similarity  ← selected (~900× faster)
        │       └─► Needleman-Wunsch Global Alignment  ← benchmark only
        │
        ├─► Feature Engineering (1051-dim)
        │       ├─► ProtBERT-BFD embeddings  [1024-dim]
        │       ├─► Physicochemical features [4-dim]
        │       ├─► Amino acid composition   [20-dim]
        │       └─► CTD structural groups    [3-dim]
        │
        ├─► StandardScaler + Balanced Class Weights
        │
        ├─► Train / Test Split (80% / 20%, stratified)
        │
        ├─► Model Training
        │       ├─► XGBoost          ← best (98.4% accuracy)
        │       ├─► Random Forest
        │       ├─► KNN
        │       └─► Voting Ensemble
        │
        └─► Sequence Generation & Prediction
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UniProt Swiss-Prot (reviewed) |
| Total Sequences | 10,544 |
| Gene Classes | 16 |
| Sequences per Gene | 659 (perfectly balanced) |
| Training Set | 8,435 (80%) |
| Test Set | 2,109 (20%) |
| Feature Dimensions | 1,051 |

---

## 🤖 Model Results

| Model | Accuracy | F1 Weighted | F1 Macro |
|-------|----------|-------------|----------|
| **XGBoost ⭐** | **98.40%** | **0.9840** | **0.9840** |
| KNN | 98.30% | 0.9830 | 0.9830 |
| Voting Ensemble | 98.30% | 0.9840 | 0.9830 |
| Random Forest | 98.00% | 0.9800 | 0.9800 |

**Best model:** XGBoost (`multi:softprob`, n_estimators=300, max_depth=6, lr=0.1)

---

## 🚀 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/af-gene-classifier.git
cd af-gene-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `transformers` are optional but recommended for full ProtBERT inference. Without them, the app falls back to mean-embedding mode (still functional — predictions driven by the 27 handcrafted features).

### 3. Place model artifacts

Ensure the following files are in the same directory as `app.py`:

```
app.py
best_model.pkl
scaler.pkl
label_encoder.pkl
protbert_embeddings.npy   ← optional but recommended
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📁 File Structure

```
af-gene-classifier/
├── app.py                     # Streamlit web application
├── AF_Gene_Classifier.ipynb   # Training notebook (end-to-end pipeline)
├── best_model.pkl             # Trained XGBoost classifier
├── scaler.pkl                 # Fitted StandardScaler
├── label_encoder.pkl          # LabelEncoder for gene classes
├── protbert_embeddings.npy    # Pre-computed ProtBERT embeddings (10544 × 1024)
└── requirements.txt           # Python dependencies
```

---

## 🔬 Inference Modes

The app automatically selects the best available inference mode:

| Mode | Requires | Description |
|------|----------|-------------|
| **Full ML** | `best_model.pkl` + `torch` + `transformers` | Live ProtBERT embedding + XGBoost prediction |
| **Fast** | `best_model.pkl` + `protbert_embeddings.npy` | Mean-embedding proxy + XGBoost (no GPU needed) |
| **Demo** | Nothing | Handcrafted biophysical heuristics (no ML artifacts) |

---

## 🧰 Tech Stack

- **Embeddings:** [ProtBERT-BFD](https://huggingface.co/Rostlab/prot_bert_bfd) (`Rostlab/prot_bert_bfd`) — pre-trained on 2.1B sequences from the BFD database
- **Classifier:** XGBoost with histogram-based gradient boosting
- **App framework:** Streamlit
- **Feature engineering:** scikit-learn, NumPy
- **Visualisation:** Plotly

---

## ⚙️ Notebook: Training from Scratch

To reproduce the full pipeline, open `AF_Gene_Classifier.ipynb` on Kaggle:

1. Upload `AF_clean_sequences.csv` and `combined_features.csv` as a Kaggle Dataset
2. Attach it to the notebook via **Add Data → Your Datasets**
3. Update `INPUT_DIR` in Cell 3 to match your dataset path
4. Run all cells — GPU is recommended for ProtBERT embedding extraction

---

## 📝 Input Guidelines

- Use the **standard 20-letter amino acid code**: `A C D E F G H I K L M N P Q R S T V W Y`
- Minimum **20** amino acids, maximum **5000**
- Spaces, numbers, and non-standard characters are automatically stripped
- For best results, use full-length or near-full-length sequences from UniProt

---

## 📄 License

This project is released for academic and research use. Sequence data sourced from [UniProt Swiss-Prot](https://www.uniprot.org/) (reviewed, manually annotated).

---

## 🙏 Acknowledgements

- [Rostlab](https://rostlab.org/) for the ProtBERT-BFD model
- [UniProt Consortium](https://www.uniprot.org/) for curated protein sequence data
- [XGBoost](https://xgboost.readthedocs.io/) and [scikit-learn](https://scikit-learn.org/) teams
