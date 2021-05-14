import time
from nlp_data_process import *
import numpy as np
import torch
from sklearn import metrics
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from rnn_classify import *
import os
from tools import config
from nlp_test import evaluate


# %%
def train(config, model, trains_iter, dev_iter):

    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(logdir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epoch):
        print('Epoch [{}/{}]'.format(epoch+1, config.num_epoch))
        for ind, (trains, labels) in tqdm(enumerate(trains_iter)):
            trains = trains.to(config.device)
            labels = labels.to(config.device)
            # mask = mask.to(config.device)
            # tokens = tokens.to(config.device)
            out = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 1000 == 0 and total_batch != 0:
                # 统计1000batch 然后输出在训练集和验证机（所有batch）上的结果
                true = labels.data.cpu()
                pre = torch.max(out.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, pre)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    if not os.path.exists(config.save_path):
                        open(config.save_path, 'w', encoding="utf-8")
                        torch.save(model.state_dict(), config.save_path)
                    else:
                        torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_diff = round(time.time()-start_time, 4)
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.2} Train Acc: {2:>6.2%},  ' \
                      'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(
                    msg.format(total_batch, loss.item(), train_acc, dev_loss,
                               dev_acc, time_diff, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print('没有提升了')
                flag = True
                break
        if flag:
            break
    writer.close()


# %%
if __name__ == '__main__':

    vocal = []
    with open('./data/vocab.txt', 'r', encoding='utf-8') as f:
        for i in f.readlines():
            vocal.append(i.strip())
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次都一样

    # 数据集的定义
    train_dataset = NewsData("./data/train.txt", model="train")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=collect_fn,
                                  shuffle=True)
    dev_dataset = NewsData("./data/dev.txt", model="valid")
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.batch_size,
                                collate_fn=collect_fn,
                                shuffle=True)
    model = RnnClassifier(len(vocal))
    train(config, model, train_dataloader, dev_dataloader)
