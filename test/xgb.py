import xgboost as xgb
import numpy as np
import csv
import pickle

# 数据加载与预处理
def load_data(file_path):
    data = []
    labels = []
    feature_mapping = {}  # 用于存储非数值特征的映射
    attack_mapping = {}   # 用于存储攻击类型的映射
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            processed_row = []
            for i, value in enumerate(row[:-2]):  # 遍历特征列
                try:
                    processed_row.append(float(value))  # 尝试将值转换为浮点数
                except ValueError:
                    if i not in feature_mapping:
                        feature_mapping[i] = {}
                    if value not in feature_mapping[i]:
                        feature_mapping[i][value] = len(feature_mapping[i])
                    processed_row.append(feature_mapping[i][value])
            attack_type = row[-2]
            if attack_type not in attack_mapping:
                attack_mapping[attack_type] = len(attack_mapping)
            label = attack_mapping[attack_type]  # 攻击类型的标签
            data.append(processed_row)
            labels.append(label)
    return np.array(data), np.array(labels), attack_mapping

# 主程序
if __name__ == "__main__":
    # 加载训练和测试数据
    train_data, train_labels, attack_mapping = load_data("./data/NSL-KDD-DataSet/KDDTrain+.csv")
    test_data, test_labels, _ = load_data("./data/NSL-KDD-DataSet/KDDTest+.csv")

    # 转换为 xgboost 的 DMatrix 格式
    dtrain = xgb.DMatrix(train_data, label=train_labels)
    dtest = xgb.DMatrix(test_data, label=test_labels)

    # 设置 XGBoost 参数
    params = {
        'objective': 'multi:softmax',  # 多分类任务
        'num_class': len(attack_mapping),  # 类别数量
        'max_depth': 5,
        'eta': 0.1,
        'eval_metric': 'mlogloss'
    }
    num_round = 100

    # 训练模型
    model = xgb.train(params, dtrain, num_round)

    # 保存模型到本地
    model_file = "./xgb_model.pkl"
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)
    print(f"模型已保存到 {model_file}")

    # 加载模型并进行预测
    # with open(model_file, 'rb') as file:
    #     model = pickle.load(file)
    predictions = model.predict(dtest)

    # 保存测试结果到本地
    result_file = "./xgb_test_results.csv"
    with open(result_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sample Index', 'True Label', 'Predicted Label'])
        for i, (true_label, predicted_label) in enumerate(zip(test_labels, predictions)):
            writer.writerow([i, true_label, predicted_label])
    print(f"测试结果已保存到 {result_file}")

    # 打印攻击类型映射
    print("攻击类型映射：")
    for attack, idx in attack_mapping.items():
        print(f"{idx}: {attack}")
