# DFauLo 

## Description
This repository is the official implementation of the tool **DfauLo**.

**DfauLo** is a dynamic data fault localization tool for deep neural networks (DNNs), which can locate mislabeled and noisy data in the deep learning datasets. Inspired by conventional mutation-based code fault localization, **DfauLo** generates multiple DNN model mutants of the original trained DNN model and maps the extracted features into a suspiciousness score indicating the probability of the given data being a data fault. **DfauLo** is the first dynamic data fault localization technique, prioritizing the suspected data based on user feedback and providing the generalizability to unseen data faults during training.


![overview](pictures/overviewISSTA-v1.jpg) 

[comment]: <> (## Sequence of code execution in the repository)

[comment]: <> (```mermaid)

[comment]: <> (graph LR)

[comment]: <> (cifar10_gendata --> cifar10_train;)

[comment]: <> (cifar10_train --> cifar10_Outlier;)

[comment]: <> (cifar10_train --> cifar10_Activation;)

[comment]: <> (cifar10_train --> cifar10_PreLoss;)

[comment]: <> (cifar10_Outlier --> cifar10_mutation;)

[comment]: <> (cifar10_Activation --> cifar10_mutation;)

[comment]: <> (cifar10_PreLoss --> cifar10_mutation;)

[comment]: <> (cifar10_mutation --> cifar10_DFauLo;)

[comment]: <> (```)
## Installation
`pip install -r requirements.txt`

## Usage
We prepared a complete demo running **DFauLo** on the EMNIST dataset mentioned in the paper. You can run this demo by directly executing:

`python dfaulo.py`

If you want to reproduce our experimental results:
+ You should first download our data&model via: https://pan.baidu.com/s/1i9CKNWfULaMlaNMs7QRgaw?pwd=dflo **Extract code**: `dflo`
+ **For RQ1 & RQ3:** you should check and run the `python exp_effective.py`
+ **For RQ2:** you should check and run the `python exp_ablation.py`
+ **For RQ4:** you should check and run the `python retrain.py`


## Run **DFauLo** on any Custom Datasets
You should first prepare your classification dataset and classes file in **dataset** file for data fault localization according to the following format (The root directory is the name of the dataset, the subordinate directory represents it as the training set, then each set directory is named by the class name of the dataset, and the images of the corresponding class are stored):
```python
MNIST # dataset name
|-- train # trainset
    |-- dog # class name
        |-- XX.png # corrensponding images
        |-- XX.png
        |-- ...
    |-- cat
        |-- XX.png
        |-- XX.png
        |-- ...
    |-- ...
|-- classes.json # class name and corresponding index
```
The `classes.json` file contains the corresponding index of the classes names when you are training the model. It needs to be written in the following form:
```python
{
    "dog": 0,
    "cat": 1,
    "person": 2,
     ...
}
```
Next you need to prepare a model trained on the above dataset and the model's transform, loss function in **models** file, we recommend that you save your model in the following form:
```python
transform = transforms.Compose([
    # MNIST transform
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

torch.save(
    {
        'transform': transform,
        'loss_fn': nn.CrossEntropyLoss(),
        'optimizer': "SGD"
    },
    '../dataset/mnist_model_args.pth'
)
```
Note that when training the model, using a `transform` with regularization can achieve better results for dfaulo.

In addition, you need to provide your model structure `model.py` in **models** file(coded in Pytorch form). Check out our sample files if you're not clear about it.


When having everything above ready, you can run our **DfauLo** tool as the following example:

Note: Please download the data in the above cloud disk before running it.

Run **DfauLo** on MNIST (image-classification dataset) example (dataset's fault type: RandomLabelNoise):
+ `python dfaulo.py --dataset './dataset/RandomLabelNoise/MNIST' --model './dataset/RandomLabelNoise/MNIST/LeNet1.pth' --model_name 'LeNet1' --class_path './dataset/mnist_classes.json' --image_size '(28,28,1)' --model_args './models/model_args.pth' --image_set 'train' --hook_layer 's4' --rm_ratio 0.05 --retrain_epoch 10 --retrain_bs 64`

Run **DfauLo** on AgNews (text-classification dataset) example (dataset's fault type: RandomLabelNoise):
+ `python dfaulo.py --dataset './dataset/RandomLabelNoise/AgNews' --model './dataset/RandomLabelNoise/MNIST/LSTM.pth' --model_name 'LSTM' --class_path './dataset/agnews_classes.json' --image_size 'None' --model_args './models/model_args.pth' --image_set 'train' --hook_layer 'fc1' --rm_ratio 0.05 --retrain_epoch 10 --retrain_bs 64`

Parameter explanation:

`--dataset ` : Path of dataset requiring data fault localization.

`--model ` : Corresponding trained model.

`--model_name`: Name of your Model.

`--class_path ` : Path of the class file.

`--image_size ` : Model input image size in (w,h,d) format.

`--model_args ` : Other parameters of the model mentioned above.

`--image_set ` : The associated set of datasets you need to perform **DfauLo**, usually 'train' or 'test'.

`--hook_layer ` : The name of the model representation layer from which you need to extract features (We recommend using the second last linear layer).

`--rm_ratio ` : Proportion of data to be removed when performing a mutation.

`--retrain_epoch ` : Epoch for model fine-tuning with DfauLo tool.

`--retrain_bs ` : BarchSize for model fine-tuning with DfauLo tool.


## Example
Some `DfauLo.py` results on benchmark **MNIST** are shown below :
<div><table frame=void>	<!--用了<div>进行封装-->
	<tr>
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./pictures/DFaLo_offline_result/2_label_4.png"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        	label: 4	<!--标题1-->
        </center></div></td>    
     	<td><div><center>	<!--第二张图片-->
    		<img src="./pictures/DFaLo_offline_result/8_label_7.png"
                 height="120"/>	
    		<br>
    		label: 7
        </center></div></td>
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./pictures/DFaLo_offline_result/15_label_6.png"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        	label: 6	<!--标题1-->
        </center></div></td> 
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./pictures/DFaLo_offline_result/21_label_7.png"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        	label: 7	<!--标题1-->
        </center></div></td> 
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./pictures/DFaLo_offline_result/45_label_1.png"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        	label: 1	<!--标题1-->
        </center></div></td> 
	</tr>
</table></div>







