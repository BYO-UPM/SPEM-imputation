# \# Imputation of Missing Data in Smooth Pursuit Eye Movements Using Deep Learning Approaches

# 

# \[!\[License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

# \[!\[Paper](https://img.shields.io/badge/arXiv-2506.00545-b31b1b.svg)](https://arxiv.org/abs/2506.00545)

# \[!\[PyTorch](https://img.shields.io/badge/PyTorch-CUDA\_11.8-EE4C2C.svg)](https://pytorch.org/)

# 

# \*\*Authors:\*\* Mehdi Bejani, Guillermo Pérez-de-Arenaza-Pozo, Julián D. Arias-Londoño, Juan I. Godino-Llorente  

# 

# \## Overview

# 

# This repository contains the official code implementation for the paper:

# > \*\*Imputation of Missing Data in Smooth Pursuit Eye Movements Using a Deep Learning Approach\*\* > \*arXiv Preprint \[arXiv:2506.00545](https://arxiv.org/abs/2506.00545)\*

# 

# Missing data is a critical issue in time series analysis, particularly in biomedical sequences like Smooth Pursuit Eye Movements (SPEM), which often contain gaps due to eye blinks and tracking losses. This project introduces a novel imputation framework that leverages state-of-the-art deep learning models—including \*\*SAITS\*\*, \*\*BRITS\*\*, and \*\*CSDI\*\* (implemented via \[PyPOTS](https://github.com/WenjieDu/PyPOTS))—combined with a custom-made \*\*Autoencoder\*\*. 

# 

# The performance of these deep learning models is rigorously compared against classical imputation methods such as \*\*KNN\*\*, \*\*PCHIP\*\*, and \*\*SSA\*\*, demonstrating significant improvements in reducing error metrics (MAE, MRE, RMSE) while preserving frequency domain characteristics.

# 

# \---

# 

# \## Repository Structure

# 

# ```text

# SPEM-Imputation/

# │

# ├── README.md                      # Project documentation and instructions

# ├── requirements.txt               # Library dependencies (PyTorch CUDA 11.8)

# ├── .gitignore                     # Git ignore file

# │

# ├── Autoencoder.py                 # Core autoencoder training and evaluation script

# ├── DL\_models.py                   # Script for testing and evaluating the DL models

# ├── pipline\_all\_BRITS.py           # Imputation pipeline using the BRITS model

# ├── pipline\_all\_CSDI.py            # Imputation pipeline using the CSDI model

# ├── pipline\_all\_SAITS.py           # Imputation pipeline using the SAITS model

# │

# ├── instances\_dict\_train.json      # Dictionary mapping for training instances

# ├── instances\_dict\_test.json       # Dictionary mapping for testing instances

# │

# ├── Data\_in/                       # Directory for datasets (See 'Data Preparation')

# │   ├── Train/                     # Training data arrays (.npy)

# │   └── Test/                      # Testing data arrays (.npy)

# │

# └── weights/                       # Directory for saved model weights

# &#x20;   ├── csdi\_weights/              # Saved PyPOTS weights for CSDI

# &#x20;   ├── saits\_weights/             # Saved PyPOTS weights for SAITS

# &#x20;   ├── brits\_weights/             # Saved PyPOTS weights for BRITS

# &#x20;   └── autoencoder\_weights/       # Saved PyTorch weights for the Autoencoder

# ```

# 

# \---

# 

# \## Getting Started

# 

# \### 1. Prerequisites \& Installation

# This project requires Python 3.8+ and uses PyTorch configured for \*\*CUDA 11.8\*\*.

# 

# Clone the repository and install the required dependencies:

# ```bash

# git clone \[https://github.com/BYO-UPM/SPEM-imputation.git](https://github.com/BYO-UPM/SPEM-imputation.git)

# cd SPEM-imputation

# 

# \# Install requirements (This will pull PyTorch for cu118 as specified in requirements.txt)

# pip install -r requirements.txt

# ```

# 

# \### 2. Data Preparation

# > \*\*Note on Data Availability:\*\* Due to the clinical nature of the dataset (comprising 172 Parkinsonian patients and healthy controls) and pending publication, the raw patient data arrays are not publicly included in this repository.

# 

# To run the code, you must place your preprocessed `.npy` data files into the `Data\_in/Train/` and `Data\_in/Test/` directories. The scripts expect files such as `SmoothPur\_1\_4.npy`, `SmoothPur\_5\_8.npy`, etc., alongside the provided JSON dictionaries (`instances\_dict\_train.json` and `instances\_dict\_test.json`).

# 

# \### 3. Usage

# 

# You can run the individual pipelines for each model to compute the imputations and calculate metrics against classical methods (KNN, PCHIP, SSA):

# 

# \* \*\*Run the SAITS pipeline:\*\*

# &#x20; ```bash

# &#x20; python pipline\_all\_SAITS.py

# &#x20; ```

# \* \*\*Run the BRITS pipeline:\*\*

# &#x20; ```bash

# &#x20; python pipline\_all\_BRITS.py

# &#x20; ```

# \* \*\*Run the CSDI pipeline:\*\*

# &#x20; ```bash

# &#x20; python pipline\_all\_CSDI.py

# &#x20; ```

# 

# To train or refine the custom Autoencoder using your imputed data, run:

# ```bash

# python Autoencoder.py

# ```

# 

# To evaluate the specific Deep Learning models dynamically (you can toggle `RUN\_SAITS`, `RUN\_BRITS`, and `RUN\_CSDI` inside the script), run:

# ```bash

# python DL\_models.py

# ```

# 

# \---

# 

# \## Citation

# 

# If you find this code or research helpful in your work, please cite our paper:

# 

# ```bibtex

# @misc{bejani2025imputationmissingdatasmooth,

# &#x20;     title={Imputation of Missing Data in Smooth Pursuit Eye Movements Using a Deep Learning Approach}, 

# &#x20;     author={Mehdi Bejani and Guillermo Perez-de-Arenaza-Pozo and Julián D. Arias-Londoño and Juan I. Godino-LLorente},

# &#x20;     year={2025},

# &#x20;     eprint={2506.00545},

# &#x20;     archivePrefix={arXiv},

# &#x20;     primaryClass={cs.LG},

# &#x20;     url={\[https://arxiv.org/abs/2506.00545](https://arxiv.org/abs/2506.00545)}, 

# }

# ```

# 

# \---

# 

# \## License

# 

# This repository is licensed under \[CC-BY-NC 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

# 

# \---

# 

# \## Contact

# 

# For questions or further information, please contact:  

# \*\*mehdi.bejani@upm.es\*\*

# ```


