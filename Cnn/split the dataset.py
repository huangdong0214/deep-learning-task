import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ===================== 1. 定义基础路径（核心：解决路径拼接问题） =====================
# 脚本所在目录（TextCnn）的绝对路径（避免相对路径混乱）
base_dir = os.path.dirname(os.path.abspath(__file__))
# 数据集根目录：TextCnn/dataset
dataset_root = os.path.join(base_dir, "dataset")
# 子文件夹路径
train_dir = os.path.join(dataset_root, "train")
test_dir = os.path.join(dataset_root, "test")
val_dir = os.path.join(dataset_root, "val")


# ===================== 2. 自动创建目录结构 =====================
def create_dirs():
    dirs = [dataset_root, train_dir, test_dir, val_dir]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"成功创建目录: {dir_path}")
        else:
            print(f"目录已存在: {dir_path}")


# ===================== 3. 读取并检查CSV文件 =====================
def load_data():
    # CSV文件路径（请确认ChnSentiCorp_htl_all.csv在TextCnn/dataset下）
    csv_path = os.path.join(dataset_root, "ChnSentiCorp_htl_all.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"未找到CSV文件！\n"
            f"请确保ChnSentiCorp_htl_all.csv文件在以下路径：\n  {csv_path}\n"
            f"当前脚本所在目录：{base_dir}"
        )

    df = pd.read_csv(csv_path)
    # 检查列数是否符合预期（避免数据格式错误）
    if df.shape[1] < 2:
        raise ValueError(f"CSV文件列数异常！预期至少2列（label, review），实际{df.shape[1]}列")

    label = df.iloc[:, 0]
    review = df.iloc[:, 1]
    print("数据读取成功！总数据量：", len(df))
    return label, review


# ===================== 4. 划分数据并保存到对应文件夹 =====================
def split_and_save():
    # 创建目录
    create_dirs()

    # 读取数据
    label, review = load_data()

    # 第一步：划分训练集（80%）和临时集（20%）
    train_label, temp_label, train_review, temp_review = train_test_split(
        label, review, test_size=0.2, random_state=0, stratify=label  # stratify：保证标签分布一致
    )

    # 第二步：划分测试集（10%）和验证集（10%）
    test_label, val_label, test_review, val_review = train_test_split(
        temp_label, temp_review, test_size=0.5, random_state=0, stratify=temp_label
    )

    # 定义保存函数（避免重复代码）
    def save_data(save_dir, label_data, review_data, prefix):
        save_path = os.path.join(save_dir, f"{prefix}.csv")
        # 合并标签和文本为DataFrame
        save_df = pd.DataFrame({
            "label": label_data.reset_index(drop=True),
            "review": review_data.reset_index(drop=True)
        })
        save_df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"{prefix}集已保存：{save_path}（数据量：{len(save_df)}）")

    # 保存训练集、测试集、验证集
    save_data(train_dir, train_label, train_review, "train")
    save_data(test_dir, test_label, test_review, "test")
    save_data(val_dir, val_label, val_review, "val")

    print("\n数据划分完成！各数据集数量：")
    print(f"训练集：{len(train_label)}条（80%）")
    print(f"测试集：{len(test_label)}条（10%）")
    print(f"验证集：{len(val_label)}条（10%）")


# ===================== 执行主函数 =====================
if __name__ == "__main__":
    try:
        split_and_save()
        print("\n✅ 所有操作完成！数据已保存到：")
        print(f"  训练集：{train_dir}/train.csv")
        print(f"  测试集：{test_dir}/test.csv")
        print(f"  验证集：{val_dir}/val.csv")
    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")
