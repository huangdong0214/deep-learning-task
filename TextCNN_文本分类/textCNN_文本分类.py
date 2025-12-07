import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd


# 读取训练集 / 测试集数据
def read_data(train_or_test, num=None):
    file_path = os.path.join("..", "data", train_or_test + ".csv")
    df = pd.read_csv(file_path, encoding='utf-8')
    texts = df["review"].fillna("").astype(str).tolist()
    labels = df["label"].fillna("").astype(str).tolist()

    if num is None:
        return texts, labels
    else:
        return texts[:num], labels[:num]


# 构建词表 built_corpus
def built_corpus(train_texts, embedding_num):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}  # 0留给填充字符，1留给未知字符
    for text in train_texts:
        for word in text:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))  # 如果字符已存在字典中则取原索引，否则分配当前字典长度作为新索引
    return word_2_index, nn.Embedding(len(word_2_index), embedding_num)


class TextDataset(Dataset):
    def __init__(self, all_text, all_labels, word_2_index, max_len):
        self.all_text = all_text
        self.all_labels = all_labels
        self.word_2_index = word_2_index
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.all_text[index][:self.max_len]
        label = int(self.all_labels[index])

        text_idx = [self.word_2_index.get(i, 1) for i in text]  # 将每个字符映射为索引，若不存在则使用 1（即 <UNK>）
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))  # 若不足 max_len，用 0（<PAD>）补齐，使长度等于 max_len

        text_idx = torch.LongTensor(text_idx).unsqueeze(dim=0)

        return text_idx, label

    def __len__(self):
        return len(self.all_text)


class Block(nn.Module):
    def __init__(self, kernel_s, embedding_num, max_len, hidden_num):
        super().__init__()
        # 卷积 → ReLU → 最大池化
        self.cnn = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(kernel_s, embedding_num))
        self.act = nn.ReLU()
        self.mxp = nn.MaxPool1d(kernel_size=(max_len - kernel_s + 1))

    def forward(self, batch_emb):
        c = self.cnn.forward(batch_emb)
        a = self.act.forward(c)
        a = a.squeeze(dim=-1)
        m = self.mxp.forward(a)
        m = m.squeeze(dim=-1)

        return m


class TextCNNModel(nn.Module):
    def __init__(self, emb_matrix, max_len, class_num, hidden_num):
        super().__init__()
        self.emb_num = emb_matrix.weight.shape[1]
        self.block1 = Block(2, self.emb_num, max_len, hidden_num)
        self.block2 = Block(3, self.emb_num, max_len, hidden_num)
        self.block3 = Block(4, self.emb_num, max_len, hidden_num)

        self.emb_matrix = emb_matrix

        self.classifier = nn.Linear(hidden_num * 3, class_num)  # 定义一个从 hidden_num × 3 维输入 到 class_num 维输出 的全连接分类层
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, batch_idx, batch_label=None):
        batch_emb = self.emb_matrix(batch_idx)
        b1_result = self.block1.forward(batch_emb)
        b2_result = self.block2.forward(batch_emb)
        b3_result = self.block3.forward(batch_emb)

        feature = torch.cat([b1_result, b2_result, b3_result], dim=1)
        pre = self.classifier(feature)

        # 根据是否传入标签（batch_label），决定模型执行 “训练模式” 还是 “预测模式”
        if batch_label is not None:
            loss = self.loss_fun(pre, batch_label)
            return loss
        else:
            return torch.argmax(pre, dim=-1)


if __name__ == "__main__":
    # 读数据
    train_text, train_label = read_data("train")
    dev_text, dev_label = read_data("dev")

    # 超参数设置
    embedding_num = 200  # 词向量维度
    max_len = 120  # 文本截断长度
    batch_size = 128  # 批次大小
    epoch = 20  # 训练轮数
    hidden_num = 100
    class_num = len(set(train_label))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建词表和 embedding
    word_2_index, words_embedding = built_corpus(train_text, embedding_num)

    train_dataset = TextDataset(train_text, train_label, word_2_index, max_len)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)

    dev_dataset = TextDataset(dev_text, dev_label, word_2_index, max_len)
    dev_loader = DataLoader(dev_dataset, batch_size, shuffle=False)

    model = TextCNNModel(words_embedding, max_len, class_num, hidden_num).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for e in range(epoch):
        for batch_idx, batch_label in train_loader:
            batch_idx = batch_idx.to(device)
            batch_label = batch_label.to(device)
            loss = model.forward(batch_idx, batch_label)  ## 传了标签 → 训练模式
            loss.backward()
            opt.step()
            opt.zero_grad()

            print(f"loss:{loss:.3f}")

            right_num = 0
            for batch_idx, batch_label in dev_loader:
                batch_idx = batch_idx.to(device)
                batch_label = batch_label.to(device)
                pre = model.forward(batch_idx)  # 没传标签 → 预测模式
                right_num += int(torch.sum(pre == batch_label))

            print(f"acc = {right_num / len(dev_text) * 100:2f}%")

    torch.save(model.state_dict(), "textcnn_model.pth")
    print("模型已保存为 textcnn_model.pth")
