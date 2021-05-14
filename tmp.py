import pickle

import torch
import torch.nn.functional as F
import numpy as np
import time
from tools import config
from sklearn import metrics
from nlp_test import evaluate
from nlp_data_process import NewsData, collect_fn

# id2label = {value: key for key, value in config.label2id.items()}
# print(id2label)
with open('result.pkl', 'rb') as f:
    ss = pickle.load(f)
print(ss)

# 判断 txt 文件是否存在，不存在就创建，然后写入文本
# msg = "世界你好\n111111111世界你好"
# configFile = "C:\\Users\\Administrator\\Desktop\\tmp.txt"
# if os.path.exists(configFile) == False:  #
#     open(configFile, 'w', encoding="utf-8")
#     # file.write(msg)
#     # file.close()
# else:
#     with open(configFile,'w', encoding="utf-8") as f:
#         f.write(msg)
#         f.close()
