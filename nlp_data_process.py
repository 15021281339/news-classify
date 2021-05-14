from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import torch
import jieba
from dictionary import *


# %%
class NewsData(Dataset):
    """
    自定义数据集
    """

    def __init__(self, data_path, model='train'):
        super(NewsData, self).__init__()
        is_test = True if model == "test" else False
        self.label_map = {item: index for index, item in enumerate(self.label_list)}  # 标签数字化
        self.example = self.read_file(data_path, is_test)

    def read_file(self, data_path, is_test):
        examples = []
        word2id = {}
        with open('./data/vocab.txt', 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                word2id[line.strip()] = idx
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if is_test:
                    text = line.strip()
                    words = jieba.lcut(text)
                    words = [word for word in words if word.strip() != '']
                    token_id = [word2id[word] for word in words if word in word2id]
                    examples.append((token_id,))
                else:
                    text, label = line.strip('\n').split('\t')
                    words = jieba.lcut(text)
                    words = [word for word in words if word.strip() != '']
                    token_id = [word2id[word] for word in words if word in word2id]
                    label = self.label_map[label]
                    examples.append((token_id, label))
        return examples

    def __getitem__(self, item):

        return self.example[item]

    def __len__(self):

        return len(self.example)

    @property
    def label_list(self):
        return ['财经', '彩票', '房产', '股票', '家居', '教育', '科技', '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']


def padding(indice, max_length, pad_idx=0):
    """
    填充每个batch的句子长度
    """
    pad_indice = [item + [pad_idx] * (max_length - len(item)) for item in indice]
    return torch.tensor(pad_indice)


# %%
def collect_fn(batch):
    """
    :动态返回tensor，padding填补句子长度
    :return: 每个batch的所有 id_tensor和label
    """

    token_ids = [data[0] for data in batch]
    max_length = max([len(t) for t in token_ids])
    labels = torch.tensor([data[1] for data in batch])

    token_ids_padded = padding(token_ids, max_length)
    return token_ids_padded, labels


# %%
def pre_collect_fn(batch):

    token_ids = [data[0] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_ids_padded = padding(token_ids, max_length)
    return token_ids_padded
