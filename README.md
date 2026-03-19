

# Imputation of Missing Data in Smooth Pursuit Eye Movements Using a Deep Learning Approach

**Author:** Mehdi Bejani   
**License:** CC-BY-NC-ND 4.0 International  
**arXiv Preprint:** [arXiv:2506.00545](https://arxiv.org/abs/2506.00545

## Overview

This repository accompanies the paper:

> **Imputation of Missing Data in Smooth Pursuit Eye Movements Using a Deep Learning Approach**  
> *M Bejani, G Perez-de-Arenaza-Pozo, JD Arias-Londoño, JI Godino-Llorente*
> arXiv Preprint [arXiv:2506.00545](https://arxiv.org/abs/2506.00545) 
 

The code uses a [SAITS](https://github.com/WenjieDu/SAITS) based model for the imputation of smooth pursuit eye movement data. The project compares the performance of the [SAITS](https://github.com/WenjieDu/SAITS) model against various other imputation methods, including **KNN**, **PCHIP**, and **SSA**. 

---

## Repository Structure

The repository is organized into the following main directories:

- **pipeline_all/**: Contains the main code for fitting the [SAITS](https://github.com/WenjieDu/SAITS) model to the data, comparing it with other imputation methods, and computing various performance metrics.
- **data_preprocessing/**: Includes all preprocessing scripts necessary for preparing the data for use in the pipeline_all directory.
- **Autoencoder/**: Contains the code for training and testing the autoencoder, with the best weights saved for use in the pipeline_all directory.

---

## Citation

If you use this work, please cite:

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

For questions, please contact:  
**mehdi.bejani@upm.es**

---

**Keywords:** Eye movements | Smooth pursuit | Imputation | SAITS | Refinement Atoencoder 

