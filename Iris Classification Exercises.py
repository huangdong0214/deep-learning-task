import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

#1.读取数据
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'  #定义 Iris 数据集在 UCI 上的原始 CSV 文件 URL（没有列名）。
col_names = ['sepal_length','sepal_width','petal_length','petal_width','class']  #指定 CSV 文件的列名（原始文件无表头）。
df = pd.read_csv(url, names=col_names)  #用 pandas 读取远程 CSV，并用上面指定的列名创建 DataFrame df。

#2.特征与标签
X = df[['sepal_length','sepal_width','petal_length','petal_width']].values.astype(np.float32)  #从 DataFrame 中选取 4 个特征列，转为 numpy 数组并转换数据类型为 float32。
y = df['class'].values  #把类别列（字符串，如 "Iris-setosa"）提取为 numpy 数组。
le = LabelEncoder()  #创建 LabelEncoder 实例，用于把字符串标签编码为数值标签。
y = le.fit_transform(y)  #用 LabelEncoder 把字符串类别映射为整数（0、1、2），并把结果赋回 y。

#3.划分训练/测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# test_size=0.2 表示把数据按80%/20%随机分为训练/测试集；random_state=42 固定随机种子以便复现；stratify=y 保持类别比例。

#4.标准化
scaler = StandardScaler()  #创建 StandardScaler 实例，用于对输入特征做标准化（均值 0，方差 1）。
X_train = scaler.fit_transform(X_train)  #在训练集上计算均值和方差并应用到训练集。
X_test = scaler.transform(X_test)  #使用训练集的均值和方差对测试集做标准化。

#5.转成 PyTorch 的 TensorDataset 和 DataLoader
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))  #把 numpy 数组转换为 PyTorch 张量，并封装为 TensorDataset（样本成对）。
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  #用 DataLoader 包装训练集，batch_size=16，shuffle=True 表示每轮打乱数据顺序。
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

#6.定义模型
class IrisNet(nn.Module):  #定义一个名为 IrisNet 的神经网络类，继承 nn.Module（所有 PyTorch 模型的基类）。
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):  #构造函数，接受输入维度、隐藏层维度和输出维度。
        super(IrisNet, self).__init__()  #调用父类构造函数，初始化模块内部状态。
        self.net = nn.Sequential(  #用 nn.Sequential 串联模块，便于按顺序定义多层网络。
            nn.Linear(input_dim, hidden_dim),  #第一层线性全连接：输入维度 -> 隐藏维度。
            nn.ReLU(),  #ReLU 激活函数，添加非线性。
            nn.Linear(hidden_dim, hidden_dim),  #第二层线性：隐藏维度 -> 隐藏维度（增加网络容量）。
            nn.ReLU(),  #再次使用 ReLU 激活。
            nn.Linear(hidden_dim, output_dim)  #输出层线性变换：隐藏维度 -> 输出类别数（logits）。
        )
    def forward(self, x):  #定义前向传播方法，输入张量 x。
        return self.net(x)  # 输入 x 依次通过 self.net（上面定义的层序列），返回未经过 softmax 的 logits。

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #判断是否有可用 GPU（CUDA），若有则使用 'cuda'，否则使用 'cpu'。
model = IrisNet().to(device)  #创建 IrisNet 实例并把模型参数移动到指定设备（GPU 或 CPU）。

#7.定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  #交叉熵损失函数，适用于多分类；它接收 logits（未归一化输出）和整数标签。
optimizer = optim.Adam(model.parameters(), lr=0.01)  #使用 Adam 优化器优化模型参数，学习率 lr=0.01。

#8.训练模型并记录指标
num_epochs = 50  #设置训练轮数为 50。
train_losses = []  # 保存每轮训练损失
test_accuracies = []  # 保存每轮测试准确率

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0  #初始化累积损失，用于计算该 epoch 的平均损失。
    for xb, yb in train_loader:  #遍历训练数据加载器，xb 为一个 batch 的特征，yb 为对应的标签。
        xb, yb = xb.to(device), yb.to(device)  #把当前 batch 的数据移动到与模型相同的设备（GPU/CPU）。
        optimizer.zero_grad()  #在反向传播前将梯度清零（PyTorch 会累积梯度）。
        outputs = model(xb)  #前向传播：把输入 xb 传入模型，得到输出 logits（shape: batch_size x num_classes）。
        loss = criterion(outputs, yb)  #计算该 batch 的交叉熵损失，outputs 是 logits，yb 是长整型标签。
        loss.backward()  #反向传播，计算参数的梯度（依据当前 loss）。
        optimizer.step()  #优化器根据计算出的梯度更新模型参数。
        running_loss += loss.item() * xb.size(0)  #把该 batch 的损失（标量）乘以样本数累加，用于后续计算平均损失。loss.item() 取出 Python float。
    epoch_loss = running_loss / len(train_loader.dataset)#把本轮在所有 batch 累积的总损失除以训练集样本数，得到该轮的平均训练损失（epoch loss）。
    train_losses.append(epoch_loss)  # 记录训练损失

    # 在每轮结束时评估测试集准确率
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            _, preds = torch.max(outputs, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    accuracy = correct / total
    test_accuracies.append(accuracy)  # 记录测试准确率

    print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Test Acc: {accuracy:.4f}')

#9.绘制 Loss 与 Accuracy 曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, 'g-o', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()