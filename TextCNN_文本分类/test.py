import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------- 复用原有核心类/函数----------------------
def read_data(train_or_test, num=None):
    file_path = os.path.join("..", "data", train_or_test + ".csv")
    df = pd.read_csv(file_path, encoding='utf-8')
    texts = df["review"].fillna("").astype(str).tolist()
    labels = df["label"].fillna("").astype(str).tolist()
    if num is None:
        return texts, labels
    else:
        return texts[:num], labels[:num]

def built_corpus(train_texts, embedding_num):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    for text in train_texts:
        for word in text:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    return word_2_index, nn.Embedding(len(word_2_index), embedding_num)

class Block(nn.Module):
    def __init__(self, kernel_s, embedding_num, max_len, hidden_num):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(kernel_s, embedding_num))
        self.act = nn.ReLU()
        self.mxp = nn.MaxPool1d(kernel_size=(max_len - kernel_s + 1))

    def forward(self, batch_emb):
        c = self.cnn(batch_emb)
        a = self.act(c)
        a = a.squeeze(dim=-1)
        m = self.mxp(a)
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
        self.classifier = nn.Linear(hidden_num * 3, class_num)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, batch_idx, batch_label=None):
        batch_emb = self.emb_matrix(batch_idx)
        b1_result = self.block1(batch_emb)
        b2_result = self.block2(batch_emb)
        b3_result = self.block3(batch_emb)
        feature = torch.cat([b1_result, b2_result, b3_result], dim=1)
        pre = self.classifier(feature)
        if batch_label is not None:
            loss = self.loss_fun(pre, batch_label)
            return loss
        else:
            return torch.argmax(pre, dim=-1)

# ---------------------- 封装单条文本预测函数 ----------------------
def init_model():
    """初始化模型（仅执行一次，避免重复加载）"""
    # 配置参数（和训练时完全一致）
    embedding_num = 200
    max_len = 120
    hidden_num = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 重建词表
    train_text, _ = read_data("train")
    word_2_index, words_embedding = built_corpus(train_text, embedding_num)
    class_num = len(set(read_data("train")[1]))

    # 加载模型
    model = TextCNNModel(words_embedding, max_len, class_num, hidden_num).to(device)
    model.load_state_dict(
        torch.load("textcnn_model.pth", map_location=device, weights_only=True)
    )
    model.eval()  # 评估模式

    return model, word_2_index, max_len, device

def predict_single_text(model, word_2_index, max_len, device, text):
    """
    单条文本预测
    :param model: 初始化后的模型
    :param word_2_index: 词表
    :param max_len: 文本最大长度
    :param device: 设备（CPU/GPU）
    :param text: 输入的文本字符串
    :return: 预测标签（整数）
    """
    # 1. 文本预处理（和训练时一致）
    text = text[:max_len]  # 截断到max_len
    text_idx = [word_2_index.get(char, 1) for char in text]  # 字符转索引，未知字符用<UNK>
    text_idx = text_idx + [0] * (max_len - len(text_idx))  # 填充到max_len
    text_idx = torch.LongTensor(text_idx).unsqueeze(dim=0).unsqueeze(dim=0)  # 增加batch和channel维度

    # 2. 执行预测
    with torch.no_grad():
        text_idx = text_idx.to(device)
        pred = model.forward(text_idx)  # 预测
        pred_label = pred.cpu().item()  # 转成整数

    return pred_label

# ---------------------- 交互式输入预测 ----------------------
if __name__ == "__main__":
    # 初始化模型（仅加载一次）
    print("正在加载模型...")
    model, word_2_index, max_len, device = init_model()
    print("模型加载完成！\n")

    # 交互式输入
    print("===== 文本分类预测工具 =====")
    print("输入 'exit' 或 'quit' 退出程序")
    while True:
        # 获取用户输入
        user_input = input("\n请输入要预测的文本：").strip()

        # 退出条件
        if user_input.lower() in ["exit", "quit"]:
            print("程序已退出！")
            break

        # 空输入处理
        if not user_input:
            print("错误：输入文本不能为空！")
            continue

        # 预测并输出结果
        pred_label = predict_single_text(model, word_2_index, max_len, device, user_input)
        # 可选：将数字标签映射为中文（根据你的数据集调整，比如0=负面，1=正面）
        label_map = {0: "负面", 1: "正面"}  # 请根据实际标签含义修改
        print(f"\n预测结果：{pred_label}（{label_map.get(pred_label, '未知')}）")
