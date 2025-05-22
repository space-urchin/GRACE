 
# GRACE: Uncertainty-Guided Coreset Selection for Active Class-Incremental Learning

**GRACE** (GReedy Aquisition via Coreset and Estimated uncertainty) is a framework for active class-incremental learning that selects informative samples using density-based uncertainty measures that guide a greedy coreset selection approach. It is designed for scenarios where new classes appear over time and labeling budgets are limited.

GRACE provides two strategies for estimate uncertainty to guide coreset selection:
- **KDE** (Kernel Density Estimation)
- **GMM** (Gaussian Mixture Modeling)

---

## 🛠️ Setup

We recommend using **conda** to create an isolated environment.

### Step-by-step:

1. Clone the repository and navigate to the project root.
2. Create a conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate myenv
```

This will install all required dependencies including PyTorch, NumPy, pandas, scikit-learn, etc.

---

## 📁 Folder Structure

```
project-root/
│
├── data/               # Contains datasets (ACI, CIC, UNSW, etc.)
│
├── info/               # The paper draft can be found here
│
├── outputs/            # Output directory for saved models and results
│
├── scripts/            # All core code files
│   ├── evaluate.py             # Evaluation logic
│   ├── grace.py                # GRACE: our proposed active learning method (KDE/GMM)
│   ├── main.py                 # Main script to run experiments
│   ├── model.py                # MLP model definition
│   ├── preparedata.py          # Data loaders for ACI, CIC, UNSW
│   └── utils.py                # Utility functions (metrics, seed setting)
```

---

## 🚀 Running the Code

To execute the full pipeline, simply run:

```bash
python scripts/main.py
```

This script will:
- Load a dataset (the dataset, streamsize, old and new classes are defined at the beginning of the script)
- Initialize and train an MLP model
- Perform active learning using the GRACE strategy
- Save results and model checkpoints to `outputs/`

- You can configure which dataset to use and what method (`kde` or `gmm`) in `main.py`.
- You can change acquisition batch size, stream size, or other hyperparameters in `main.py`.

---

## 🔍 GRACE: Active Learning Strategy

The main contribution is in `scripts/grace.py`, which implements:

- **KDE-based density estimation**:  
  Computes uncertainty based on local sample density using Epanechnikov kernel.
  ```python
  method='kde'
  ```

- **GMM-based density estimation**:  
  Fits a class-conditional GMM and uses likelihood for uncertainty estimation.
  ```python
  method='gmm'
  ```

You can alternate between methods by passing the `method` argument when calling the GRACE selection function.

---

## 🧠 Model

A simple fully-connected **MLP** is defined in `scripts/model.py`. You can modify it to:
- Add layers or activation functions
- Integrate other architectures
- Use pretrained models

The current model is lightweight and designed for tabular datasets.

---

## 📊 Datasets

GRACE is an off-the-shelf approach that is model and data-agnostic.

You will find existing implementation for the following intrusion detection datasets:

- **ACI-IOT-2023** – IoT Anomaly Classification
- **CICIDS2017** – Canadian Institute for Cybersecurity Intrusion Detection
- **UNSW-NB15** – Network security attack classification

Each dataset:
- Has specific preprocessing logic in `scripts/preparedata.py`
- Must be placed in the `data/` directory
- Is compatible with the active learning loop in `main.py`
- To use a different dataset, add its loading logic to `preparedata.py`.

---

## 🧪 Evaluation

Evaluation metrics used in `scripts/evaluate.py` include:
- **Old Class F1 Score**: Performance on known classes
- **New Class F1 Score**: Performance on new classes encountered during training
- **Overall F1 Score**
- **Accuracy**

These metrics are saved to `outputs/` for each experiment.

---


