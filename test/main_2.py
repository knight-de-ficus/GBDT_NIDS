import csv
import numpy as np
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

# 手动实现决策树回归器
class DecisionTreeRegressor:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(set(y)) == 1:
            return np.mean(y)
        feature, threshold = self._find_best_split(X, y)
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_indices], y[left_indices], depth + 1),
            'right': self._build_tree(X[right_indices], y[right_indices], depth + 1)
        }

    def _find_best_split(self, X, y):
        best_feature, best_threshold, best_loss = None, None, float('inf')
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                loss = self._calculate_loss(y[left_indices], y[right_indices])
                if loss < best_loss:
                    best_feature, best_threshold, best_loss = feature, threshold, loss
        return best_feature, best_threshold

    def _calculate_loss(self, left_y, right_y):
        left_loss = np.var(left_y) * len(left_y)
        right_loss = np.var(right_y) * len(right_y)
        return left_loss + right_loss

    def _predict_single(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])

# 自定义GBDT实现
class GBDT:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # 初始化残差为目标值（确保为浮点类型）
        residual = y.astype(float).copy()
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            self.trees.append(tree)
            # 更新残差
            predictions = tree.predict(X)
            residual -= self.learning_rate * predictions
            # 输出中间过程信息
            print(f"第{i + 1}棵树训练完成，残差均值: {np.mean(residual):.4f}")

    def predict(self, X):
        # 累加所有树的预测值
        pred = np.zeros(X.shape[0])
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        # 转换为二分类结果
        return (pred > 0.5).astype(int)

    def save_model(self, file_path):
        """保存模型到本地"""
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(file_path):
        """从本地加载模型"""
        with open(file_path, 'rb') as file:
            return pickle.load(file)

# 手动实现准确率、精确度和召回率计算
def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def precision_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positive = np.sum(y_pred == 1)
    return true_positive / predicted_positive if predicted_positive > 0 else 0

def recall_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    actual_positive = np.sum(y_true == 1)
    return true_positive / actual_positive if actual_positive > 0 else 0

# 主程序
if __name__ == "__main__":
    # 加载训练和测试数据
    train_data, train_labels, attack_mapping = load_data("./data/NSL-KDD-DataSet/KDDTrain+.csv")
    test_data, test_labels, _ = load_data("./data/NSL-KDD-DataSet/KDDTest+.csv")

    # 初始化并训练GBDT模型
    gbdt = GBDT(n_estimators=50, learning_rate=0.1, max_depth=3)
    gbdt.fit(train_data, train_labels)

    # 保存模型到本地
    model_file = "./gbdt_model.pkl"
    gbdt.save_model(model_file)
    print(f"模型已保存到 {model_file}")

    # 加载模型并进行预测
    gbdt = GBDT.load_model(model_file)
    predictions = gbdt.predict(test_data)

    # 保存测试结果到本地
    result_file = "./test_results.csv"
    with open(result_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sample Index', 'True Label', 'Predicted Label'])  # 写入表头
        for i, (true_label, predicted_label) in enumerate(zip(test_labels, predictions)):
            writer.writerow([i, true_label, predicted_label])  # 写入每个样本的结果
    print(f"测试结果已保存到 {result_file}")

    # 打印攻击类型映射
    print("攻击类型映射：")
    for attack, idx in attack_mapping.items():
        print(f"{idx}: {attack}")
