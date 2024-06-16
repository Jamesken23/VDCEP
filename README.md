# VDCEP
This repo is a paper on Python implementation: **A vulnerability detection framework by focusing on critical execution paths**. This paper designs a new deep learning-based framework, named VDCEP, that aims to utilize critical code execution paths to improve the performance of detecting software vulnerabilities (i.e., smart contract vulnerabilities).


# Datasets

Following prior work [Qian et al., 2023](https://dl.acm.org/doi/10.1145/3543507.3583367), we use four vulnerability datasets, i.e., reentrancy (RE), timestamp dependence (TD), integer overflow/underflow (OF), and delegatecall (DE), as the benchmark datasets. These four vulnerability datasets all come from a recently-released and large-scale smart contract dataset [Qian et al., 2023](https://dl.acm.org/doi/10.1145/3543507.3583367), which comprises 42,910 real-world smart contracts collected from the Ethereum platform [Thomas et al., 2020](https://ieeexplore.ieee.org/document/9284023). After obtaining the above vulnerability datasets, we manually check the correctness of the labeling results of  these smart contracts based on the labeling strategy in [Qian et al., 2023](https://dl.acm.org/doi/10.1145/3543507.3583367). Our work focuses on the RE, TD, OF, and DE vulnerabilities, since they possess the typical characteristics of Ethereum smart contract vulnerabilities.

 Further instructions on the dataset can be found on [Smart-Contract-Dataset](https://github.com/Messi-Q/Smart-Contract-Dataset), which is constantly being updated to provide more details.
 

# Required Packages
> - Python 3.8+
> - Pytorch 1.7.0
> - numpy 1.19.5
> - scikit-learn 1.3.0
> - Cuda 11.0

# Running
You can run the VDCEP model by: "python main.py"

If you want to get the feature weights for each execution path, you can check out line 119 (i.e., path_weights) of code in "Network/vdcep_model.py".

# Tools
For a smart contract source code, you can use this tool [SourceGraphExtractor](https://github.com/Messi-Q/SourceGraphExtractor) to generate a code structure graph.

If you want to learn more about how to deconstruct a control flow graph into a code execution path, you can refer to this work [2023 TSE EPVD](https://ieeexplore.ieee.org/document/10153647).

Following prior works [AME](https://www.ijcai.org/proceedings/2021/379) and [CGE](https://ieeexplore.ieee.org/abstract/document/9477066), we conclude vulnerability-related code statements for four types of smart contract vulnerabilities as heuristic rules.

# Reference
This work has been accepted by the journal of Information and Software Technology (IST). You can cite this paper by:
> @article{CHENG2024107517,  
>  &nbsp; &nbsp; &nbsp; title={A vulnerability detection framework by focusing on critical execution paths},  
>  &nbsp; &nbsp; &nbsp; author={Cheng, Jianxin and Chen, Yizhou and Cao, Yongzhi and Wang, Hanpin},  
>  &nbsp; &nbsp; &nbsp; journal={Information and Software Technology},  
>  &nbsp; &nbsp; &nbsp; pages={107517},  
>  &nbsp; &nbsp; &nbsp; year={2024},  
>  &nbsp; &nbsp; &nbsp; publisher={Elsevier}  
> }
