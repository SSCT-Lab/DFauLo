# DFauLo 

## Description
This repository is the official implementation of the tool **DfauLo**.

**DfauLo** is a dynamic data fault localization tool for deep neural networks (DNNs), which can locate mislabeled and noisy data in the deep learning datasets. Inspired by conventional mutation-based code fault localization, **DfauLo** generates multiple DNN model mutants of the original trained DNN model and maps the extracted features into a suspiciousness score indicating the probability of the given data being a data fault. **DfauLo** is the first dynamic data fault localization technique, prioritizing the suspected data based on user feedback and providing the generalizability to unseen data faults during training.

The tool works by taking *a trained model and its optimizer and lr_scheduler (if available)*, and *a (classification) dataset* for data fault localization in the required format. It then generates multiple DNN model mutants, extracts features from these mutants, and maps the extracted features into a suspiciousness score indicating the probability of the given data being a data fault. The user can then prioritize the suspected data for manual checking and correction to improve the data quality with minimal effort. The tool can effectively locate different types of data faults and improve the model performance by correcting the located faults.

The implementation of **DfauLo** as a tool allows for the automatic location of data faults hidden in deep learning datasets. It is applicable for various types of deep learning datasets and can be used to validate the data quality for DNNs. The source code for the tool is available for use and can be customized according to specific requirements.


![overview](./pictures/overviewISSTA-v1.jpg) 

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

## Requirement
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
Next you need to prepare a model trained on the above dataset (note that you need to provide the entire model and not just the parameters of the model) and the model's 
optimizer, transform, loss function and lr_scheduler (if available) in **models** file, we recommend that you save your model in the following form:
```python
torch.save({
    "epoch": epoch,  # optional
    "optimizer": optimizer,
    "transform": transform,
    "loss_fn": criterion,
    "lr_scheduler": lr_scheduler,  # optional
    "acc": 100 * correct / total  # optional
}, './models/model_args.pth')

torch.save(model, './models/model.pth') # save the entire model
```
Note that when training the model, using a `transform` with regularization can achieve better results for dfaulo.

In addition, you need to provide your model structure `model.py` in **models** file(coded in Pytorch form). Check out our sample files if you're not clear about it.

## Usage
When having everything above ready, you can run our **DfauLo** tool as the following example:
+ `python dfaulo.py --dataset './dataset/mnist' --model './models/model.pth' --image_size '(28,28,1)' --model_args './models/model_args.pth' --image_set 'train' --hook_layer 's4' --rm_ratio 0.05 --top_ratio 0.01 --retrain_epoch 10 --retrain_bs 64
--slice_num 1
 `

Parameter explanation:

`--dataset ` : Path of dataset requiring data fault localization.

`--model ` : Corresponding trained model.

`--img_size ` : Model input image size in (w,h,d) format.

`--model_args ` : Other parameters of the model mentioned above.

`--image_set ` : The associated set of datasets you need to perform **DfauLo**, usually 'train' or 'test'.

`--hook_layer ` : The name of the model representation layer from which you need to extract features (We recommend using the second last linear layer).

`--rm_ratio ` : Proportion of data to be removed when performing a mutation.

`--top_ratio ` : Percentage of fault data needed.

`--retrain_epoch 10 ` : Epoch for model fine-tuning with DfauLo tool.

`--retrain_bs ` : BarchSize for model fine-tuning with DfauLo tool.

`--slice_num ` : The number of slices performed on the dataset. If your dataset is large, we recommend running **DfauLo** with this parameter to split the dataset into multiple subsets equally. This helps prevent memory overflows and speeds up **DfauLo** in some cases, but also reduces the accuracy.

The tool will generate the **located_image_list.txt** file in the subdirectory of the dataset input. with the paths to the images we located on your dataset that contain faults.

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







