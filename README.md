# Imputation of Missing Data in Smooth Pursuit Eye Movements Using Deep Learning Approaches

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Paper](https://img.shields.io/badge/arXiv-2506.00545-b31b1b.svg)](https://arxiv.org/abs/2506.00545)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_11.8-EE4C2C.svg)](https://pytorch.org/)

**Authors:** Mehdi Bejani, Guillermo Pérez-de-Arenaza-Pozo, Julián D. Arias-Londoño, Juan I. Godino-Llorente

---

## Overview

This repository contains the official code implementation for the paper:

> **Imputation of Missing Data in Smooth Pursuit Eye Movements Using a Deep Learning Approach**
> *arXiv Preprint [arXiv:2506.00545](https://arxiv.org/abs/2506.00545)*

Missing data is a critical issue in time series analysis, particularly in biomedical sequences like Smooth Pursuit Eye Movements (SPEM), which often contain gaps due to eye blinks and tracking losses. This project introduces a novel imputation framework that leverages state-of-the-art deep learning models—including **SAITS**, **BRITS**, and **CSDI** (implemented via [PyPOTS](https://github.com/WenjieDu/PyPOTS))—combined with a custom-made **Autoencoder**.

The performance of these deep learning models is rigorously compared against classical imputation methods such as **KNN**, **PCHIP**, and **SSA**, demonstrating significant improvements in reducing error metrics (MAE, MRE, RMSE) while preserving frequency domain characteristics.

---

## Repository Structure

```text
SPEM-Imputation/
│
├── README.md                      # Project documentation and instructions
├── requirements.txt               # Library dependencies (PyTorch CUDA 11.8)
├── .gitignore                     # Git ignore file
│
├── Autoencoder.py                 # Core autoencoder training and evaluation script
├── DL_models.py                   # Script for testing and evaluating the DL models
├── pipline_all_BRITS.py           # Imputation pipeline using the BRITS model
├── pipline_all_CSDI.py            # Imputation pipeline using the CSDI model
├── pipline_all_SAITS.py           # Imputation pipeline using the SAITS model
│
├── instances_dict_train.json      # Dictionary mapping for training instances
├── instances_dict_test.json       # Dictionary mapping for testing instances
│
├── Data_in/                       # Directory for datasets (See 'Data Preparation')
│   ├── Train/                     # Training data arrays (.npy)
│   └── Test/                      # Testing data arrays (.npy)
│
└── weights/                       # Directory for saved model weights
    ├── csdi_weights/              # Saved PyPOTS weights for CSDI
    ├── saits_weights/             # Saved PyPOTS weights for SAITS
    ├── brits_weights/             # Saved PyPOTS weights for BRITS
    └── autoencoder_weights/       # Saved PyTorch weights for the Autoencoder
```

---

## Getting Started

### 1. Prerequisites & Installation

This project requires Python 3.8+ and uses PyTorch configured for **CUDA 11.8**.

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/BYO-UPM/SPEM-imputation.git
cd SPEM-imputation

# Install requirements (this will pull PyTorch for cu118 as specified in requirements.txt)
pip install -r requirements.txt
```

### 2. Data Preparation

> **Note on Data Availability:** Due to the clinical nature of the dataset (comprising 172 Parkinsonian patients and healthy controls) and pending publication, the raw patient data arrays are not publicly included in this repository.

To run the code, place your preprocessed `.npy` data files into the `Data_in/Train/` and `Data_in/Test/` directories. The scripts expect files such as `SmoothPur_1_4.npy`, `SmoothPur_5_8.npy`, etc., alongside the provided JSON dictionaries (`instances_dict_train.json` and `instances_dict_test.json`).

### 3. Usage

Run the individual pipelines for each model to compute imputations and calculate metrics against classical methods (KNN, PCHIP, SSA):

* **Run the SAITS pipeline:**
  ```bash
  python pipline_all_SAITS.py
  ```
* **Run the BRITS pipeline:**
  ```bash
  python pipline_all_BRITS.py
  ```
* **Run the CSDI pipeline:**
  ```bash
  python pipline_all_CSDI.py
  ```

To train or refine the custom Autoencoder using your imputed data, run:

```bash
python Autoencoder.py
```

To evaluate deep learning models dynamically (toggle `RUN_SAITS`, `RUN_BRITS`, and `RUN_CSDI` inside the script), run:

```bash
python DL_models.py
```

---

## Citation

If you find this code or research helpful in your work, please cite our paper:

```bibtex
@misc{bejani2025imputationmissingdatasmooth,
      title={Imputation of Missing Data in Smooth Pursuit Eye Movements Using a Deep Learning Approach},
      author={Mehdi Bejani and Guillermo Perez-de-Arenaza-Pozo and Julián D. Arias-Londoño and Juan I. Godino-LLorente},
      year={2025},
      eprint={2506.00545},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.00545},
}
```

---

## License

This repository is licensed under [CC-BY-NC 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

---

## Contact

For questions or further information, please contact:
**mehdi.bejani@upm.es**
