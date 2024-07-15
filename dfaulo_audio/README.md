
# 将Dfaulo用于Audio分类问题


## 直接运行
先下载数据：

链接：https://pan.baidu.com/s/1gTXwV2llGIgVU6PvxtcMaw?pwd=dfad 
提取码：dfad 

运行`python exp_effective.py`

运行结果：

![alt text](image-1.png)


## *脏模型训练流程：

### 1. 缺陷生成
`audio_fault_gen.py`用于生成缺陷，现支持5%的RandomLabelNoise，以下用rln表示。


### 2. 训练Audio分类模型：

请先阅读AudioClassification-Pytorch的README。

正确设置`tdnn_fea_extra.yml`并改变`extract_features.py`中对应参数，然后运行，以提取特征。

提取完后得到文件:
- `dataset/features_rln/` 存放用于训练的特征
- `dataset/train_list_rln_features.txt`
- `dataset/test_list_features.txt`

训练得到缺陷模型

`python train.py`

![alt text](image.png)

