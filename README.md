# DataDebugging
![markdown picture](./pictures/overview.Png) 
## Sequence of code execution in the repository
```mermaid
graph LR
cifar10_gendata --> cifar10_train;
cifar10_train --> cifar10_Outlier;
cifar10_train --> cifar10_Activation;
cifar10_train --> cifar10_PreLoss;
cifar10_Outlier --> cifar10_mutation;
cifar10_Activation --> cifar10_mutation;
cifar10_PreLoss --> cifar10_mutation;
cifar10_mutation --> cifar10_DFauLo;
```
## Installation
`pip install -r requirements.txt`

## Usage
We prepare a demo for DFauLo:
+ `python demo.py`

You should use **this link** to download our noisy data or raw data into the `demodata` folder.


