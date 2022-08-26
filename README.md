# DataDebugging
![overview](./pictures/overview.Png) 
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

You should first prepare the **MNIST** dataset classified in the following format and save it in the **demodata** folder:

```
MNIST
|-- train
    |-- 0
        |-- XX.png
        |-- XX.png
        |-- ...
    |-- 1
        |-- XX.png
        |-- XX.png
        |-- ...
    |-- ...
```
Some `demo.py` results are shown below:


![overview](./demodata/DFaLo_offline_result/0_label_9.png)(kave)

<center>
	<img src="./demodata/DFaLo_offline_result/0_label_9.png" width="30%" />
	&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
	<img src="./demodata/DFaLo_offline_result/0_label_9.png" width="30%" />
	<br/>
	<font color="AAAAAA">001.jpg</font>
	&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
	&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
	&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
	<font color="AAAAAA">002.jpg</font>
</center>
<br/>

