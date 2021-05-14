import collections
import jieba
from itertools import chain
import os
import re
import pandas as pd
from string import punctuation
from collections import Counter
from tqdm import tqdm
import pandas as pd


# %% 构建词汇表 （粒度为word）
class Dictionary(object):
    """构建字典"""
    def __init__(self, train_path, valid_path, test_path, min_count=5):

        self.train_path = train_path
        self.test_path = test_path
        self.valid_path = valid_path
        self.min_count = min_count

    def build_dictionary(self):

        all_words = []
        files = [self.train_path, self.valid_path, self.test_path]
        for file_name in files:
            with open(file_name, 'r', encoding='utf-8') as f:
                for line in f:
                    if file_name in [self.train_path, self.valid_path]:
                        text, label = line.strip().split('\t')
                    elif file_name == self.test_path:
                        text = line.strip()
                    else:
                        continue
                    words = jieba.lcut(text)
                    words = [word for word in words if word.strip() != '']
                    all_words.append(words)
        self._build_dictionary(all_words, './data/vocab.txt')

    def _build_dictionary(self, all_words, file_path):

        vocab_words = []
        words = list(chain(*all_words))  # 多维合并为一维
        words_vocab = collections.Counter(words).most_common()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('[UNK]\n[PAD]\n')
            vocab_words.append('[UNK]\n[PAD]\n')
            for word, num in words_vocab:
                if num < self.min_count:
                    continue
                f.write(word + '\n')
                vocab_words.append(word + '\n')
        self.word2id = dict(zip(vocab_words, range(len(vocab_words))))

    def indexer(self, word):
        """根据词查询id"""
        try:
            idx = self.word2id[word]
        except:
            idx = self.word2id['[UNK]']
        return idx


# %%
if __name__ == '__main__':
    train_path = './data/train.txt'
    valid_path = './data/dev.txt'
    test_path = './data/test.txt'

    dic = Dictionary(train_path, valid_path, test_path)
    dic.build_dictionary()
