import os  # 引入os模块，用于文件路径操作
import numpy as np  # 引入numpy用于数值处理
import torch  # 引入PyTorch库
import torch.nn as nn  # 导入神经网络模块
from torch.utils.data import Dataset, DataLoader  # 引入Dataset和DataLoader用于数据加载
from tqdm import tqdm  # 进度条库
import pandas as pd  # 读取csv数据


# 读取训练集 / 测试集数据
def read_data(train_or_test, num=None):  # 定义数据读取函数，参数为train/dev/test
    file_path = os.path.join("..", "data", train_or_test + ".csv")  # 拼接数据路径
    df = pd.read_csv(file_path, encoding='utf-8')  # 按UTF-8编码读取CSV文件
    texts = df["review"].fillna("").astype(str).tolist()  # 获取文本列，空值替换为""，转换为字符串列表
    labels = df["label"].fillna("").astype(str).tolist()  # 获取标签列，处理方式同上

    if num is None:  # 若num未指定，返回全部数据
        return texts, labels
    else:
        return texts[:num], labels[:num]  # 否则返回前num条


# 构建词表 built_corpus
def built_corpus(train_texts, embedding_num):  # 输入训练文本和词向量维度
    word_2_index = {"<PAD>": 0, "<UNK>": 1}  # 初始化词典，0是填充符，1是未登录词
    for text in train_texts:  # 遍历所有文本
        for word in text:  # 遍历文本内每个字符（因为逐字符）
            word_2_index[word] = word_2_index.get(word, len(word_2_index))  # 若已存在返回原索引，否则分配新索引
    return word_2_index, nn.Embedding(len(word_2_index), embedding_num)  # 返回词典和embedding层


class TextDataset(Dataset):  # 定义自定义数据集
    def __init__(self, all_text, all_labels, word_2_index, max_len):  # 保存必要参数
        self.all_text = all_text  # 保存全部文本
        self.all_labels = all_labels  # 保存全部标签
        self.word_2_index = word_2_index  # 保存词典
        self.max_len = max_len  # 最大文本长度

    def __getitem__(self, index):  # 获取一条数据
        text = self.all_text[index][:self.max_len]  # 截断文本到max_len
        label = int(self.all_labels[index])  # 标签转为整数

        text_idx = [self.word_2_index.get(i, 1) for i in text]  # 转化为索引序列，未知字用1
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))  # 若长度不足，用0补齐

        text_idx = torch.LongTensor(text_idx).unsqueeze(dim=0)  # 转成Tensor，增加一个维度(卷积要求)

        return text_idx, label  # 返回文本索引与标签

    def __len__(self):  # 返回数据集大小
        return len(self.all_text)


class Block(nn.Module):  # 定义一个卷积块
    def __init__(self, kernel_s, embedding_num, max_len, hidden_num):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(kernel_s, embedding_num))  # 定义卷积层
        self.act = nn.ReLU()  # ReLU 激活
        self.mxp = nn.MaxPool1d(kernel_size=(max_len - kernel_s + 1))  # 最大池化（池化窗口根据输入序列长度而定）

    def forward(self, batch_emb):  # 前向传播
        c = self.cnn.forward(batch_emb)  # 进行卷积
        a = self.act.forward(c)  # 激活操作
        a = a.squeeze(dim=-1)  # 去掉embedding维度
        m = self.mxp.forward(a)  # 最大池化
        m = m.squeeze(dim=-1)  # 去掉pool维度

        return m  # 返回池化后的特征


class TextCNNModel(nn.Module):  # 定义TextCNN模型结构
    def __init__(self, emb_matrix, max_len, class_num, hidden_num):
        super().__init__()
        self.emb_num = emb_matrix.weight.shape[1]  # 获取词向量维度
        self.block1 = Block(2, self.emb_num, max_len, hidden_num)  # 卷积核大小2
        self.block2 = Block(3, self.emb_num, max_len, hidden_num)  # 卷积核大小3
        self.block3 = Block(4, self.emb_num, max_len, hidden_num)  # 卷积核大小4

        self.emb_matrix = emb_matrix  # 保存embedding层

        self.classifier = nn.Linear(hidden_num * 3, class_num)  # 全连接层，输入为三个block拼接
        self.loss_fun = nn.CrossEntropyLoss()  # 使用交叉熵损失

    def forward(self, batch_idx, batch_label=None):  # batch_label 决定是否训练
        batch_emb = self.emb_matrix(batch_idx)  # 获取embedding
        b1_result = self.block1.forward(batch_emb)  # 2卷积核结果
        b2_result = self.block2.forward(batch_emb)  # 3卷积核结果
        b3_result = self.block3.forward(batch_emb)  # 4卷积核结果

        feature = torch.cat([b1_result, b2_result, b3_result], dim=1)  # 拼接所有卷积特征
        pre = self.classifier(feature)  # 分类输出

        if batch_label is not None:  # 若传入标签 → 训练模式
            loss = self.loss_fun(pre, batch_label)  # 计算损失
            return loss
        else:
            return torch.argmax(pre, dim=-1)  # 预测模式 → 返回分类结果


if __name__ == "__main__":  # 主程序
    train_text, train_label = read_data("train")  # 读取训练集数据
    dev_text, dev_label = read_data("dev")  # 读取验证集

    embedding_num = 200  # 词向量维度
    max_len = 120  # 最大序列长度
    batch_size = 128  # batch大小
    epoch = 20  # 训练轮数
    hidden_num = 100  # 卷积输出通道数
    class_num = len(set(train_label))  # 类别数量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用GPU

    word_2_index, words_embedding = built_corpus(train_text, embedding_num)  # 构建词典与embedding

    train_dataset = TextDataset(train_text, train_label, word_2_index, max_len)  # 训练集Dataset
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)  # DataLoader

    dev_dataset = TextDataset(dev_text, dev_label, word_2_index, max_len)  # 验证集Dataset
    dev_loader = DataLoader(dev_dataset, batch_size, shuffle=False)  # DataLoader

    model = TextCNNModel(words_embedding, max_len, class_num, hidden_num).to(device)  # 初始化模型
    opt = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

    for e in range(epoch):  # 训练循环
        for batch_idx, batch_label in train_loader:  # 遍历训练集
            batch_idx = batch_idx.to(device)  # 数据送到GPU/CPU
            batch_label = batch_label.to(device)
            loss = model.forward(batch_idx, batch_label)  # 前向传播，带label → 返回loss
            loss.backward()  # 反向传播
            opt.step()  # 参数更新
            opt.zero_grad()  # 梯度清零

            print(f"loss:{loss:.3f}")  # 打印当前loss

            right_num = 0  # 统计正确个数
            for batch_idx, batch_label in dev_loader:  # 遍历验证集
                batch_idx = batch_idx.to(device)
                batch_label = batch_label.to(device)
                pre = model.forward(batch_idx)  # 不传label → 预测
                right_num += int(torch.sum(pre == batch_label))  # 计算预测正确数

            print(f"acc = {right_num / len(dev_text) * 100:2f}%")  # 打印准确率

    torch.save(model.state_dict(), "textcnn_model.pth")  # 保存模型参数
    print("模型已保存为 textcnn_model.pth")  # 提示保存完成

