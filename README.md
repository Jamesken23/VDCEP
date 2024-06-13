# VDCEP
This repo is a paper on Python implementation: **A vulnerability detection framework by focusing on critical execution paths**. This paper designs a new deep learning-based framework, named VDCEP, that aims to utilize critical code execution paths to improve the performance of detecting software vulnerabilities (i.e., smart contract vulnerabilities).


# Datasets

Following prior work [Qian et al., 2023](https://dl.acm.org/doi/10.1145/3543507.3583367), we use four vulnerability datasets, i.e., reentrancy (RE), timestamp dependence (TD), integer overflow/underflow (OF), and delegatecall (DE), as the benchmark datasets. These four vulnerability datasets all come from a recently-released and large-scale smart contract dataset [Qian et al., 2023](https://dl.acm.org/doi/10.1145/3543507.3583367), which comprises 42,910 real-world smart contracts collected from the Ethereum platform [Thomas et al., 2020](https://ieeexplore.ieee.org/document/9284023). After obtaining the above vulnerability datasets, we manually check the correctness of the labeling results of  these smart contracts based on the labeling strategy in [Qian et al., 2023](https://dl.acm.org/doi/10.1145/3543507.3583367). Our work focuses on the RE, TD, OF, and DE vulnerabilities, since they possess the typical characteristics of Ethereum smart contract vulnerabilities.

 Further instructions on the dataset can be found on [Smart-Contract-Dataset](https://github.com/Messi-Q/Smart-Contract-Dataset), which is constantly being updated to provide more details.
 

# Required Packages
> - Python 3.8+
> - Pytorch 1.10.0
> - numpy 1.19.5
> - scikit-learn 1.3.0
> - Cuda 11.3

