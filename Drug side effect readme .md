# Drug Side-Effect Prediction with Machine Learning

**BINF 6210 – Machine Learning for Bioinformatics | Final Project**
**Author:** Ser (UNC Charlotte, M.S. Bioinformatics & Genomics)

---

## 1. Overview

Adverse drug reactions (ADRs) are a major cause of drug failure in clinical trials and of drug withdrawals from the market. If side effects can be predicted from a drug's **chemical structure** alone, risky compounds can be flagged early in development.

This project frames side-effect prediction as a **multi-label classification** problem. Given a drug molecule (as a SMILES string), the models predict which side-effect categories it is associated with. Models range from classical ML baselines to deep learning models: MLP, CNN/LSTM on SMILES, Graph Neural Networks, and a fine-tuned pretrained chemical language model.

### Research questions

1. How well can molecular structure alone predict drug side effects?
2. Do graph-based and sequence-based deep models outperform classical models on fingerprints?
3. Does transfer learning from a pretrained chemistry model help on this small dataset?
4. Which structural features drive the predictions, and how do dataset biases affect the results?

---

## 2. Dataset

| Dataset                         | Description                                                                                      | Use                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| **SIDER (MoleculeNet version)** | ~1,427 marketed drugs × 27 side-effect categories (MedDRA System Organ Classes), with SMILES     | Primary dataset (multi-label, 27 tasks)              |
| **SIDER 4.1 (raw)**             | ~1,430 drugs, ~5,800 side-effect terms, ~140k drug–side effect pairs; some frequency information | Extension: fine-grained labels, frequency regression |
| **PubChem**                     | Chemical structures (SMILES) by compound ID                                                      | Mapping raw SIDER IDs to structures                  |

Example side-effect categories: _Hepatobiliary disorders, Cardiac disorders, Psychiatric disorders, Skin and subcutaneous tissue disorders, Nervous system disorders._

---

## 3. Course Topics Covered

| Lectures | Topic                                             | Where it appears in the project                                                                   |
| -------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| L1–2     | Intro to ML / environment setup                   | Problem framing; conda env, Jupyter, GPU setup                                                    |
| L3–4     | Data preprocessing                                | SMILES validation and standardization, fingerprints, descriptors, class imbalance, scaffold split |
| L5–6     | Linear regression (analytical + gradient descent) | Regression baseline: predicting side-effect frequency or a molecular property (e.g., logP)        |
| L7–10    | Classification                                    | Logistic regression, Random Forest, SVM, XGBoost; multi-label metrics                             |
| L11–12   | Unsupervised learning                             | PCA / UMAP of chemical space; k-means / hierarchical clustering of drugs                          |
| L13–14   | Neural networks                                   | Multi-label MLP on fingerprints                                                                   |
| L16–17   | CNN                                               | 1D CNN on one-hot encoded SMILES                                                                  |
| L18–21   | Sequential models                                 | LSTM/GRU and a small Transformer on SMILES tokens                                                 |
| L22–23   | Graph neural networks                             | GCN / GIN / GAT on molecular graphs (**main model**)                                              |
| L24–25   | Transfer learning                                 | Fine-tuning pretrained ChemBERTa (and/or a pretrained GNN)                                        |
| L26      | Reinforcement learning                            | Discussion / future work: RL-guided design of safer molecules                                     |
| L27      | Generative AI                                     | Optional: small SMILES VAE or LLM-based label extraction (discussion)                             |
| L28      | Bias in biomedical data                           | Label imbalance, reporting bias, scaffold vs. random split, per-category error analysis           |

---

## 4. Methods

### 4.1 Preprocessing

- Parse and standardize SMILES with **RDKit** (remove salts, neutralize, canonicalize); drop invalid molecules.
- Remove duplicate drugs; check label prevalence for each category.
- **Features:**
  - Morgan (ECFP4) fingerprints, 2048 bits
  - MACCS keys (166 bits)
  - RDKit 2D descriptors (MolWt, LogP, TPSA, HBD/HBA, rotatable bonds, …), standardized
  - Tokenized SMILES (sequence models)
  - Molecular graphs: atoms = nodes, bonds = edges (GNNs)
- **Splits:** 80/10/10 with a **scaffold split** (Bemis–Murcko) as the main setting, plus a random split for comparison.
- **Imbalance:** class weights / `pos_weight` in BCE loss; report PR-AUC alongside ROC-AUC.

### 4.2 Models

| #   | Model                                                           | Input               | Lecture |
| --- | --------------------------------------------------------------- | ------------------- | ------- |
| 0   | Linear regression (closed-form + gradient descent from scratch) | Descriptors         | L5–6    |
| 1   | Logistic regression (one-vs-rest)                               | ECFP4               | L7      |
| 2   | Random Forest / XGBoost                                         | ECFP4 + descriptors | L8–10   |
| 3   | SVM (RBF / Tanimoto kernel)                                     | ECFP4               | L8–10   |
| 4   | MLP (multi-label, sigmoid outputs)                              | ECFP4               | L13–14  |
| 5   | 1D CNN                                                          | One-hot SMILES      | L16–17  |
| 6   | BiLSTM / Transformer encoder                                    | SMILES tokens       | L18–21  |
| 7   | GCN, GIN, GAT                                                   | Molecular graph     | L22–23  |
| 8   | ChemBERTa fine-tuned                                            | SMILES              | L24–25  |

### 4.3 Unsupervised analysis

- PCA and UMAP of fingerprints, colored by side-effect labels.
- Clustering drugs, then testing whether clusters are enriched for particular side effects.

### 4.4 Evaluation

- **Metrics:** macro and micro ROC-AUC, macro PR-AUC, F1 (with threshold tuning on the validation set), Hamming loss.
- **Per-category results** to show which side effects are predictable and which are not.
- 3–5 random seeds per model; report mean ± SD.
- Hyperparameters tuned on the validation set only; test set used once.

### 4.5 Interpretability

- SHAP values for tree models (which fingerprint bits/substructures matter).
- Attention or GNNExplainer for the GNN to highlight important atoms.
- Map important Morgan bits back to substructures and draw them with RDKit.

### 4.6 Bias analysis (L28)

- **Label bias:** common categories dominate; rare ones are underrepresented.
- **Reporting bias:** SIDER comes from drug labels, so "absent" does not always mean "never happens."
- **Chemical-space bias:** marketed drugs are not representative of new candidates. Compare scaffold-split vs. random-split performance.
- **Population bias:** side-effect data come from trial populations that may underrepresent some groups (discussion).

---

## 5. Repository Structure

```
drug-side-effect-prediction/
├── README.md
├── environment.yml
├── data/
│   ├── raw/                 # downloaded SIDER / PubChem files (not tracked)
│   └── processed/           # cleaned tables, features, splits
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_features.ipynb
│   ├── 03_unsupervised_analysis.ipynb
│   ├── 04_linear_regression.ipynb
│   ├── 05_classical_models.ipynb
│   ├── 06_mlp.ipynb
│   ├── 07_cnn_lstm_smiles.ipynb
│   ├── 08_gnn.ipynb
│   ├── 09_transfer_learning_chemberta.ipynb
│   ├── 10_interpretability.ipynb
│   └── 11_bias_analysis.ipynb
├── src/
│   ├── data.py              # loading, cleaning, splits
│   ├── features.py          # fingerprints, descriptors, graphs, tokenization
│   ├── models.py            # model definitions
│   ├── train.py             # training loops
│   └── evaluate.py          # metrics and plots
├── results/
│   ├── figures/
│   └── tables/
└── report/
    ├── final_report.pdf
    └── slides.pdf
```

---

## 6. Setup

```bash
conda create -n sider-ml python=3.11 -y
conda activate sider-ml

pip install numpy pandas scikit-learn xgboost matplotlib seaborn umap-learn shap jupyter
pip install rdkit
pip install torch torchvision
pip install torch_geometric
pip install transformers datasets
pip install deepchem   # optional: gives easy access to the MoleculeNet SIDER dataset
```

Loading the dataset:

```python
import deepchem as dc
tasks, (train, valid, test), _ = dc.molnet.load_sider(
    featurizer="ECFP", splitter="scaffold"
)
```

or download `sider.csv` (MoleculeNet) and load it with pandas.

Running:

```bash
jupyter notebook notebooks/
```

Run the notebooks in order (01 → 11).

---

## 7. Timeline

| Week | Tasks                                                                        |
| ---- | ---------------------------------------------------------------------------- |
| 1    | Environment setup, download data, exploratory analysis                       |
| 2    | Preprocessing, features, scaffold split, PCA/UMAP, clustering                |
| 3    | Linear regression from scratch; classical classifiers (LR, RF, SVM, XGBoost) |
| 4    | MLP, CNN, LSTM on SMILES                                                     |
| 5    | GNNs (GCN, GIN, GAT)                                                         |
| 6    | ChemBERTa fine-tuning; interpretability (SHAP, GNNExplainer)                 |
| 7    | Bias analysis, final comparison, report and presentation                     |

---

## 8. Results

_To be filled in._

| Model                  | Macro ROC-AUC | Macro PR-AUC | Micro F1 |
| ---------------------- | ------------- | ------------ | -------- |
| Logistic regression    | –             | –            | –        |
| Random Forest          | –             | –            | –        |
| XGBoost                | –             | –            | –        |
| MLP                    | –             | –            | –        |
| CNN (SMILES)           | –             | –            | –        |
| LSTM (SMILES)          | –             | –            | –        |
| GIN                    | –             | –            | –        |
| ChemBERTa (fine-tuned) | –             | –            | –        |

---

## 9. Limitations & Future Work

- SIDER is small (~1.4k drugs), so deep models may overfit; transfer learning is meant to help with this.
- Structure alone ignores dose, drug targets, patient factors and drug–drug interactions.
- **Future work:** add drug–target features (DrugBank) and build a drug–side-effect knowledge graph for link prediction. Model polypharmacy side effects (e.g., the Decagon dataset). Explore RL-based generation of molecules with lower predicted toxicity (L26–27).

---

## 10. References

1. Kuhn M, Letunic I, Jensen LJ, Bork P. _The SIDER database of drugs and side effects._ Nucleic Acids Research, 2016.
2. Wu Z, et al. _MoleculeNet: a benchmark for molecular machine learning._ Chemical Science, 2018.
3. Xu K, et al. _How Powerful are Graph Neural Networks?_ (GIN). ICLR, 2019.
4. Chithrananda S, Grand G, Ramsundar B. _ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction._ arXiv, 2020.
5. Zitnik M, Agrawal M, Leskovec J. _Modeling polypharmacy side effects with graph convolutional networks_ (Decagon). Bioinformatics, 2018.
6. Lundberg SM, Lee SI. _A Unified Approach to Interpreting Model Predictions_ (SHAP). NeurIPS, 2017.
7. RDKit: Open-source cheminformatics. https://www.rdkit.org
