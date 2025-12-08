import os  # 导入操作系统路径模块
import torch  # 导入PyTorch
import pandas as pd  # 导入pandas用于读取CSV
import torch.nn as nn  # 导入神经网络模块
from torch.utils.data import Dataset, DataLoader  # 导入数据集与数据加载器

# ---------------------- 复用原有核心类/函数----------------------
def read_data(train_or_test, num=None):  # 读取指定的数据集
    file_path = os.path.join("..", "data", train_or_test + ".csv")  # 拼接文件路径
    df = pd.read_csv(file_path, encoding='utf-8')  # 读取csv文件
    texts = df["review"].fillna("").astype(str).tolist()  # 获取文本列，空填充
    labels = df["label"].fillna("").astype(str).tolist()  # 获取标签列，空填充
    if num is None:  # 如果不限制数量
        return texts, labels  # 返回全部数据
    else:
        return texts[:num], labels[:num]  # 返回前num条数据

def built_corpus(train_texts, embedding_num):  # 构建词表与词向量层
    word_2_index = {"<PAD>": 0, "<UNK>": 1}  # 初始化词典
    for text in train_texts:  # 遍历所有文本
        for word in text:  # 遍历文本中的每个字符
            word_2_index[word] = word_2_index.get(word, len(word_2_index))  # 添加新词
    return word_2_index, nn.Embedding(len(word_2_index), embedding_num)  # 返回词表与embedding层

class Block(nn.Module):  # TextCNN 卷积块
    def __init__(self, kernel_s, embedding_num, max_len, hidden_num):
        super().__init__()  # 调用父类初始化
        self.cnn = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(kernel_s, embedding_num))  # 卷积层
        self.act = nn.ReLU()  # ReLU激活
        self.mxp = nn.MaxPool1d(kernel_size=(max_len - kernel_s + 1))  # 最大池化

    def forward(self, batch_emb):
        c = self.cnn(batch_emb)  # 卷积操作
        a = self.act(c)  # 激活
        a = a.squeeze(dim=-1)  # 去掉最后一维
        m = self.mxp(a)  # 池化
        m = m.squeeze(dim=-1)  # 再次压缩维度
        return m  # 返回结果

class TextCNNModel(nn.Module):  # TextCNN 主模型
    def __init__(self, emb_matrix, max_len, class_num, hidden_num):
        super().__init__()  # 初始化父类
        self.emb_num = emb_matrix.weight.shape[1]  # 获取embedding维度
        self.block1 = Block(2, self.emb_num, max_len, hidden_num)  # 卷积块1（kernel=2）
        self.block2 = Block(3, self.emb_num, max_len, hidden_num)  # 卷积块2（kernel=3）
        self.block3 = Block(4, self.emb_num, max_len, hidden_num)  # 卷积块3（kernel=4）
        self.emb_matrix = emb_matrix  # 词向量层
        self.classifier = nn.Linear(hidden_num * 3, class_num)  # 分类器
        self.loss_fun = nn.CrossEntropyLoss()  # 交叉熵损失

    def forward(self, batch_idx, batch_label=None):
        batch_emb = self.emb_matrix(batch_idx)  # 通过embedding层
        b1_result = self.block1(batch_emb)  # 块1输出
        b2_result = self.block2(batch_emb)  # 块2输出
        b3_result = self.block3(batch_emb)  # 块3输出
        feature = torch.cat([b1_result, b2_result, b3_result], dim=1)  # 拼接特征
        pre = self.classifier(feature)  # 分类预测
        if batch_label is not None:  # 训练模式
            loss = self.loss_fun(pre, batch_label)  # 计算损失
            return loss  # 返回损失
        else:
            return torch.argmax(pre, dim=-1)  # 返回预测类别

# ---------------------- 封装单条文本预测函数 ----------------------
def init_model():  # 初始化模型（只运行一次）
    """初始化模型（仅执行一次，避免重复加载）"""
    embedding_num = 200  # 词向量维度
    max_len = 120  # 文本最大长度
    hidden_num = 100  # 卷积输出维度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备

    train_text, _ = read_data("train")  # 读取训练集
    word_2_index, words_embedding = built_corpus(train_text, embedding_num)  # 构建词典与embedding层
    class_num = len(set(read_data("train")[1]))  # 统计类别数量

    model = TextCNNModel(words_embedding, max_len, class_num, hidden_num).to(device)  # 创建模型
    model.load_state_dict(
        torch.load("textcnn_model.pth", map_location=device, weights_only=True)  # 加载权重
    )
    model.eval()  # 进入推理模式

    return model, word_2_index, max_len, device  # 返回模型与资源

def predict_single_text(model, word_2_index, max_len, device, text):  # 单条文本预测
    text = text[:max_len]  # 截断到max_len
    text_idx = [word_2_index.get(char, 1) for char in text]  # 字符→索引
    text_idx = text_idx + [0] * (max_len - len(text_idx))  # 填充到max_len
    text_idx = torch.LongTensor(text_idx).unsqueeze(dim=0).unsqueeze(dim=0)  # 扩展维度为 batch×1×len

    with torch.no_grad():  # 推理不求梯度
        text_idx = text_idx.to(device)  # 放到设备
        pred = model.forward(text_idx)  # 执行预测
        pred_label = pred.cpu().item()  # 取出整数标签

    return pred_label  # 返回预测结果

# ---------------------- 交互式输入预测 ----------------------
if __name__ == "__main__":  # 主程序入口
    print("正在加载模型...")  # 提示
    model, word_2_index, max_len, device = init_model()  # 初始化模型
    print("模型加载完成！\n")  # 加载完成提示

    print("===== 文本分类预测工具 =====")  # 标题
    print("输入 'exit' 或 'quit' 退出程序")  # 提示
    while True:  # 循环接收输入
        user_input = input("\n请输入要预测的文本：").strip()  # 获取用户输入

        if user_input.lower() in ["exit", "quit"]:  # 判断退出
            print("程序已退出！")  # 提示
            break  # 退出循环

        if not user_input:  # 输入为空
            print("错误：输入文本不能为空！")  # 错误提示
            continue  # 忽略并继续

        pred_label = predict_single_text(model, word_2_index, max_len, device, user_input)  # 预测
        label_map = {0: "负面", 1: "正面"}  # 数字→中文标签
        print(f"\n预测结果：{pred_label}（{label_map.get(pred_label, '未知')}）")  # 输出结果

