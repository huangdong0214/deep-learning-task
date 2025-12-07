import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#1.数据读取
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
# UCI红葡萄酒质量数据集的URL链接
df = pd.read_csv(url, sep=';')  # 从URL中读取CSV文件，使用分号作为分隔符
X = df.drop('quality', axis=1).values.astype(np.float32)  # 将除“quality”列外的所有特征取出，转为float32数组
y = df['quality'].values.astype(np.float32).reshape(-1, 1)  # 提取目标列“quality”，转为float32并变为二维列向量

#2.划分数据集
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 首先将数据划分为训练集(80%)和测试集(20%)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)  # 再将训练集划分为训练集(80%)与验证集(20%)

#3.标准化
scaler_X = StandardScaler()  # 创建特征标准化器
X_train = scaler_X.fit_transform(X_train)  # 对训练特征进行标准化拟合与转换
X_val = scaler_X.transform(X_val)  # 用同样的标准对验证集进行标准化（不重新拟合）
X_test = scaler_X.transform(X_test)  # 用相同的标准对测试集进行标准化

scaler_y = StandardScaler()  # 创建标签标准化器
y_train = scaler_y.fit_transform(y_train)  # 拟合并标准化训练集标签
y_val = scaler_y.transform(y_val)  # 使用同样的标准对验证集标签进行标准化
y_test = scaler_y.transform(y_test)  # 使用同样的标准对测试集标签进行标准化

#4.转为 TensorDataset
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))  # 将训练数据转为TensorDataset对象
val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))  # 将验证数据转为TensorDataset对象
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))  # 将测试数据转为TensorDataset对象

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)  # 创建训练集数据加载器，batch大小为32，随机打乱
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)  # 创建验证集数据加载器，不打乱
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)  # 创建测试集数据加载器，不打乱

#5.定义模型
class WineNet(nn.Module):  # 定义神经网络模型WineNet，继承自nn.Module
    def __init__(self, input_dim, hidden_dim=32, output_dim=1, dropout_rate=0.3):  # 构造函数，定义网络结构参数，dropout_rate=0.3 表示模型在训练时会随机丢弃 30% 的神经元，以防止过拟合、增强泛化能力。
        super(WineNet, self).__init__()  # 调用父类构造函数
        self.net = nn.Sequential(  # 使用Sequential构建多层结构
            nn.Linear(input_dim, hidden_dim),  # 输入层到隐藏层
            nn.ReLU(),  # 激活函数ReLU
            nn.Dropout(dropout_rate),      # 添加Dropout层防止过拟合，随机丢弃部分神经元
            nn.Linear(hidden_dim, hidden_dim),  # 第二个隐藏层
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout_rate),  # 再加一个Dropout层
            nn.Linear(hidden_dim, output_dim)  # 输出层，输出一个预测值
        )
    def forward(self, x):  # 定义前向传播
        return self.net(x)  # 顺序执行Sequential中的各层

input_dim = X_train.shape[1]  # 获取输入维度（特征数）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检测是否有GPU
model = WineNet(input_dim=input_dim).to(device)  # 创建模型实例并移动到GPU或CPU

#6.定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 使用Adam优化器，设置较小学习率并加入L2正则(weight_decay)

#7.Early Stopping 设置
num_epochs = 200  # 最大训练轮数
patience = 15  # 若验证集损失15轮未提升则提前停止
best_val_loss = np.inf  # 初始化最佳验证损失为正无穷
best_epoch = 0  # 记录最佳轮次
train_losses, val_losses = [], []  # 用于记录每轮的训练与验证损失
best_model_state = None  # 存储最佳模型参数

#8.训练循环
for epoch in range(num_epochs):
    model.train()  # 启用训练模式（启用Dropout等）
    running_loss = 0.0  # 初始化本轮累计损失
    for xb, yb in train_loader:  # 遍历训练集批次
        xb, yb = xb.to(device), yb.to(device)  # 将批次数据送到GPU/CPU
        optimizer.zero_grad()  # 梯度清零
        outputs = model(xb)  # 前向传播，得到预测输出
        loss = criterion(outputs, yb)  # 计算该批次的均方误差损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        running_loss += loss.item() * xb.size(0)  # 累加当前batch的损失（乘以样本数）

    train_loss = running_loss / len(train_loader.dataset)  # 求得平均训练损失

    # 验证阶段
    model.eval()  # 切换为验证模式（关闭Dropout）
    val_running_loss = 0.0  # 初始化验证损失
    with torch.no_grad():  # 验证阶段关闭梯度计算
        for xb, yb in val_loader:  # 遍历验证集批次
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            val_loss = criterion(outputs, yb)  # 计算验证损失
            val_running_loss += val_loss.item() * xb.size(0)
    val_loss = val_running_loss / len(val_loader.dataset)  # 计算平均验证损失

    train_losses.append(train_loss)  # 记录训练损失
    val_losses.append(val_loss)  # 记录验证损失

    print(f"Epoch {epoch+1:03d}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")  # 打印当前轮结果

    # Early Stopping 逻辑
    if val_loss < best_val_loss - 1e-4:  # 若验证集损失有显著改善
        best_val_loss = val_loss  # 更新最优损失
        best_epoch = epoch  # 记录当前最佳轮数
        best_model_state = model.state_dict()  # 保存当前模型参数
    elif epoch - best_epoch > patience:  # 若超过patience轮未提升
        print(f"验证集 {patience} 轮未提升，提前停止训练。")  # 提示早停
        break  # 跳出循环提前结束训练

#9.恢复最佳模型
model.load_state_dict(best_model_state)  # 恢复训练中验证集最优模型参数
print(f"\n使用第 {best_epoch+1} 轮的最优模型进行测试评估。")  # 输出采用的最佳epoch编号

#10. 绘制训练曲线
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss (with Early Stopping)')
plt.legend()
plt.grid(True)
plt.show()

#11.测试集评估
model.eval()
preds, targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        preds.append(outputs.cpu().numpy())
        targets.append(yb.cpu().numpy())

preds = np.vstack(preds)
targets = np.vstack(targets)

# 反标准化
preds_orig = scaler_y.inverse_transform(preds)
targets_orig = scaler_y.inverse_transform(targets)

mse = mean_squared_error(targets_orig, preds_orig)  # 计算测试集均方误差
print(f"\nTest MSE: {mse:.4f}")  # 打印测试结果（模型在真实分数下的误差）

