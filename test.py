import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer

# 下载并加载AG News数据集
train_iter = AG_NEWS(split='train')

# 使用basic_english分词器
tokenizer = get_tokenizer("basic_english")

max_token_length = 0
longest_token = ""

# 遍历数据集，统计最长的token
for label, line in train_iter:
    tokens = tokenizer(line)
    for token in tokens:
        if len(token) > max_token_length:
            max_token_length = len(token)
            longest_token = token

print(f"Longest token: {longest_token}")
print(f"Length of the longest token: {max_token_length}")
