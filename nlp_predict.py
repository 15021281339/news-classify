import torch
import torch.nn.functional as F
import numpy as np
import time
import pickle
from tools import config
from sklearn import metrics
from nlp_test import evaluate
from nlp_data_process import NewsData, pre_collect_fn
from torch.utils.data import DataLoader
from rnn_classify import RnnClassifier
from tqdm import tqdm


# %%
def test(model):

    model.load_state_dict(torch.load(config.save_path))

    test_dataset = NewsData("./data/test.txt", model="test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 collate_fn=pre_collect_fn,
                                 shuffle=True)
    all_labels = []
    with torch.no_grad():
        for texts in tqdm(test_dataloader):
            texts = texts.to(config.device)
            result = model(texts)
            pre = torch.max(result.data, 1)[1].cpu().numpy()
            id2label = {value: key for key, value in config.label2id.items()}
            labels = [id2label[int(label_id)] for label_id in pre]
            all_labels = np.append(all_labels, labels)

    with open('result.pkl', 'wb') as f:
        pickle.dump(all_labels, f)


# %%
if __name__ == '__main__':
    vocal = []
    with open('./data/vocab.txt', 'r', encoding='utf-8') as f:
        for i in f.readlines():
            vocal.append(i.strip())
    np.random.seed(1)
    torch.manual_seed(1)
    model = RnnClassifier(len(vocal))
    test(model)