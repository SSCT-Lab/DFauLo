# DataDebugging
![markdown picture](./pictures/overview.Png) 
## One experimental procedure
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
We prepare a demo for DFauLo


