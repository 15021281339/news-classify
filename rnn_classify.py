import torch
import torch.nn as nn


# %% RNN模型参数
class RnnConfig:

    hidden_size = 256
    required_grad = True
    pretrain_embeddings = False  # 预训练
    pretrain_embeddings_path = None
    drop_out = 0.3
    num_class = 14
    num_layers = 2
    bidirectional = True
    embed_size = 128


# %% RNN模型结构
class RnnClassifier(nn.Module):
    """ vocab_size:字典长"""
    def __init__(self, vocab_size):  # vocab_size =len(vocal.txt)
        super(RnnClassifier, self).__init__()
        if RnnConfig.pretrain_embeddings:
            self.embeddings = nn.Embedding.from_pretrained(RnnConfig.pretrain_embeddings_path)
        else:
            self.embeddings = nn.Embedding(vocab_size, RnnConfig.embed_size)

        self.gru = nn.GRU(input_size=RnnConfig.embed_size,
                          hidden_size=RnnConfig.hidden_size,
                          batch_first=True,
                          num_layers=RnnConfig.num_layers,
                          bidirectional=RnnConfig.bidirectional)
        self.drop_out = nn.Dropout(RnnConfig.drop_out)
        self.fc = nn.Linear(2*RnnConfig.hidden_size, RnnConfig.num_class)

    def forward(self, x):
        embedding = self.embeddings(x)   # batch_size * seq_len * embed_size
        hidden, _ = self.gru(embedding)  # batch_size * seq_len * hidden_size
        hidden = self.drop_out(hidden)
        hidden = hidden[:, -1, :]  # 获取最后一层的输出
        pro = self.fc(hidden)   # 输出概率
        return pro
