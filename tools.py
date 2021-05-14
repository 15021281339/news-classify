import logging
from logging import handlers
import re
import jieba
import json
from tqdm import tqdm
# jieba.enable_parallel(4)
import pandas as pd
import time
from functools import partial, wraps
from datetime import timedelta
import torch


# %%
class config:

    num_epoch = 50
    is_cuda = False
    batch_size = 64 #128
    device = torch.device('cuda') if is_cuda else torch.device('cpu')
    # with open('./data/stop_words.txt', 'r', encoding='utf-8') as f:
    #     stopwords = [word.strip() for word in f.readlines()]
    learning_rate = 2e-5  # 学习率
    eps = 1e-8
    dropout = 0.3  # 随机失活
    require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
    save_path = './model/rnn_best_cls.pt'
    log_path = './logs'
    label2id = {'财经': 0, '彩票': 1, '房产': 2, '股票': 3, '家居': 4, '教育': 5, '科技': 6,
                '社会': 7, '时尚': 8, '时政': 9, '体育': 10, '星座': 11,'游戏': 12, '娱乐': 13}

# def rm_stop_word(wordlist):
#     """
#     删除输入数据的停用词
#     input：wordlist
#     """
#     word_list = [w for w in wordlist if w not in config.stopwords]
#     return word_list

# def query_cut(query): 切词
#     return jieba.lcut(query)
#
# def clean_symbols(text):
#     """ 特殊符号处理"""
#     text = re.sub('[0-9]+',"NUM",str(text))
#     text = re.sub('[!！]+',"!",text)
#     text = re.sub('[?？]+',"?",text)
#     text = re.sub("[a-zA-Z#$%&\'()*+,-./:;:<=>@，。★、…【】《》“”‘’'!'[\\]^_`{|}~]+", " OOV ",text)
#     return text.sub('\s+'," ",text)
#
# def process_data(data,word):
#     """数据预处理"""
#     data["sentence"] = data['title'] + data['content']
#
#     data['clean_sentence'] = data['sentence'].progress_apply(clean_symbols)
#     # data["cut_sentence"] = data['clean_sentence'].progress_apply(' '.join(query_cut))
#
#     # 标签映射到id
#     data['category_id'] = data['label'].progress_apply(lambda x: x.strip()).map(config.label2id)
#     # char粒度
#     if not word:
#         data['raw_words'] = data["cut_sentence"].apply(lambda x: " ".join(list("".join(x.split(' ')))))
#
#     return data
