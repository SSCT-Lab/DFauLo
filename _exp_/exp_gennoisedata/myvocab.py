class Vocab:
    UNK_TAG = "<UNK>"
    PAD_TAG = "<PAD>"
    PAD = 0
    UNK = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}

    def fit(self, sentence):
        # sentence=[str,str,str]
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count=None, max_count=None, max_feature=None):
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count >= min_count}
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
        if max_feature is not None:
            self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_feature])
        for word in self.count:
            self.dict[word] = len(self.dict)
        self.inverse_dict=dict(zip(self.dict.values(),self.dict.keys()))
    def transform(self, sentence, max_len=None):
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        else:
            sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
        return [self.dict.get(i, 1) for i in sentence]
    def inverse_transform(self,incides):
        return [self.inverse_dict.get(i,"<UNK>") for i in incides]
    def __len__(self):
        return len(self.dict)