import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import pandas as pd
import re
import jieba

# 忽略jieba的pkg_resources警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ===================== 1. TextCnn模型 =====================
class TextCnn(nn.Module):
    def __init__(self, num_classes):
        super(TextCnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 15, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ===================== 2. 文本数据集类 =====================
class HotelReviewDataset(Dataset):
    def __init__(self, csv_path, seq_len=60):
        # 验证文件是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV文件不存在！请检查路径：{csv_path}\n"
                f"当前工作目录：{os.getcwd()}\n"
                f"当前目录下的文件：{os.listdir(os.getcwd())}"
            )

        # 自动适配编码
        try:
            self.df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(csv_path, encoding='gbk')

        self.seq_len = seq_len

        # 自动适配文本列名
        text_cols = ['review_text', 'text', '评论', '内容']
        self.text_col = None
        for col in text_cols:
            if col in self.df.columns:
                self.text_col = col
                break
        if self.text_col is None:
            raise ValueError(f"未找到文本列！当前列名：{self.df.columns.tolist()}")

        # 文本清洗
        self.df[self.text_col] = self.df[self.text_col].apply(self._clean_text)

        # 构建词汇表
        self.word2idx, self.idx2word = self._build_vocab()
        self.vocab_size = len(self.word2idx)

        # 自动适配标签列名
        label_cols = ['label', 'sentiment', '情感']
        self.label_col = None
        for col in label_cols:
            if col in self.df.columns:
                self.label_col = col
                break
        if self.label_col is None:
            raise ValueError(f"未找到标签列！当前列名：{self.df.columns.tolist()}")

        # 标签映射（兼容中英文标签）
        try:
            label_mapping = {"negative": 0, "positive": 1, "负面": 0, "正面": 1}
            self.df[self.label_col] = self.df[self.label_col].map(label_mapping)
        except:
            # 标签已是数字，跳过
            pass

    def _clean_text(self, text):
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _build_vocab(self):
        all_words = []
        for text in self.df[self.text_col]:
            words = jieba.lcut(text)
            all_words.extend(words)

        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        special_tokens = ["<pad>", "<unk>"]
        vocab = special_tokens + [word for word, freq in word_freq.items() if freq >= 2]

        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        return word2idx, idx2word

    def _text_to_seq(self, text):
        words = jieba.lcut(text)
        unk_idx = self.word2idx["<unk>"]
        pad_idx = self.word2idx["<pad>"]
        indices = [self.word2idx.get(word, unk_idx) for word in words]

        if len(indices) > self.seq_len:
            indices = indices[:self.seq_len]
        else:
            indices += [pad_idx] * (self.seq_len - len(indices))

        return torch.tensor(indices, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx][self.text_col]
        label = self.df.iloc[idx][self.label_col]
        text_seq = self._text_to_seq(text)
        text_seq = text_seq.unsqueeze(0)  # [1, 60]
        return text_seq, torch.tensor(label, dtype=torch.long)


# ===================== 3. 训练/评估函数 =====================
def train(model, train_loader, optimizer, criterion, num_epochs, device, save_path):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"\nEpoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

        accuracy = evaluate(model, test_loader, criterion, device)
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, save_path)
            print(f"✅ Model saved! Best Accuracy: {best_acc:.2f}%")


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Evaluating", unit="batch")
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            test_bar.set_postfix(acc=100.0 * correct / total)

    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100.0 * correct / total
    print(f"📊 Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return accuracy


def save_model(model, save_path):
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


# ===================== 4. 主函数（路径已修正） =====================
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    # 修正后的数据集路径（必须确保文件存在！）
    train_csv_path = r"D:\Software\Project\PyCharmCode\Cnn\dataset\train.csv"
    test_csv_path = r"D:\Software\Project\PyCharmCode\Cnn\dataset\ChnSentiCorp_htl_all.csv"
    save_path = r"D:\Software\Project\PyCharmCode\Cnn\model_pth\best.pth"

    # 验证路径
    print(f"\n🔍 验证路径：")
    print(f"训练集路径：{train_csv_path} → {'存在' if os.path.exists(train_csv_path) else '不存在！'}")
    print(f"测试集路径：{test_csv_path} → {'存在' if os.path.exists(test_csv_path) else '不存在！'}")

    # 超参数
    num_epochs = 10
    learning_rate = 0.001
    num_classes = 2

    # 加载数据集
    try:
        train_dataset = HotelReviewDataset(train_csv_path, seq_len=60)
        test_dataset = HotelReviewDataset(test_csv_path, seq_len=60)
        print(f"\n✅ 数据集加载成功！")
        print(f"训练集条数：{len(train_dataset)}, 词汇表大小：{train_dataset.vocab_size}")
        print(f"测试集条数：{len(test_dataset)}")
    except Exception as e:
        print(f"\n❌ 数据集加载失败：{str(e)}")
        exit()

    # 数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    # 初始化模型
    model = TextCnn(num_classes=num_classes).to(device)
    print(f"\n🚀 Model initialized: {model}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    print("\n=============== Start Training ===============")
    train(model, train_loader, optimizer, criterion, num_epochs, device, save_path)

    # 最终评估
    print("\n=============== Final Evaluation ===============")
    evaluate(model, test_loader, criterion, device)

