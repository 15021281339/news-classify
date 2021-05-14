import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tools import config
from sklearn import metrics


# %% 文本分类--验证（测试）
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)  # 把每一个batch的结果存储起来最后比较所有batch的结果
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in tqdm(data_iter):
            texts = texts.to(config.device)
            labels = labels.to(config.device)
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            pre = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, pre)
    acc = metrics.accuracy_score(labels_all, predict_all)

    # if test:
    #     report = metrics.classification_report(labels_all,
    #                                            predict_all,
    #                                            target_names=config.label2id.keys(),
    #                                            digits=4)
    #     confusion = metrics.confusion_matrix(labels_all, predict_all)
    #     return acc, loss_total / len(data_iter), report, confusion

    return acc, loss_total / len(data_iter)
