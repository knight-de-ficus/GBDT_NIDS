import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt

# 数据加载与预处理
def load_data(file_path):
    data = []
    labels = []
    string_columns = {}  # 用于存储字符串列的唯一值映射

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            features = []
            for i, value in enumerate(row[:-2]):  # 遍历特征列
                try:
                    features.append(float(value))  # 尝试将值转换为浮点数
                except ValueError:
                    # 如果是字符串，进行编码处理
                    if i not in string_columns:
                        string_columns[i] = {}
                    if value not in string_columns[i]:
                        string_columns[i][value] = len(string_columns[i])
                    features.append(string_columns[i][value])
            label = 1 if row[-2] != 'normal' else 0  # 攻击为1，正常为0
            data.append(features)
            labels.append(label)

    return np.array(data), np.array(labels)

# 定义决策树模型
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, residual):
        self.tree = self._build_tree(X, residual, depth=0)

    def predict(self, X):
        return self._predict_tree(self.tree, X)

    def _build_tree(self, X, residual, depth):
        # 决策树构建逻辑
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples <= 1:
            return np.mean(residual)

        best_split = None
        min_error = float('inf')

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_residual = residual[left_mask]
                right_residual = residual[right_mask]

                left_mean = np.mean(left_residual)
                right_mean = np.mean(right_residual)
                error = (
                    np.sum((left_residual - left_mean) ** 2) +
                    np.sum((right_residual - right_mean) ** 2)
                )

                if error < min_error:
                    min_error = error
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        if best_split is None:
            return np.mean(residual)

        left_tree = self._build_tree(X[best_split['left_mask']], residual[best_split['left_mask']], depth + 1)
        right_tree = self._build_tree(X[best_split['right_mask']], residual[best_split['right_mask']], depth + 1)

        return {
            'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def _predict_tree(self, tree, X):
        # 决策树预测逻辑
        if not isinstance(tree, dict):
            return np.full(X.shape[0], tree)

        feature_idx = tree['feature_idx']
        threshold = tree['threshold']

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        predictions = np.zeros(X.shape[0])
        predictions[left_mask] = self._predict_tree(tree['left'], X[left_mask])
        predictions[right_mask] = self._predict_tree(tree['right'], X[right_mask])

        return predictions

# 定义正则化类
class Regularization:
    def __init__(self, l1=0.0, l2=0.0):
        """
        初始化正则化参数
        :param l1: L1 正则化系数
        :param l2: L2 正则化系数
        """
        self.l1 = l1
        self.l2 = l2

    def apply(self, weights):
        """
        计算正则化损失
        :param weights: 模型权重
        :return: 正则化损失
        """
        l1_loss = self.l1 * np.sum(np.abs(weights))
        l2_loss = self.l2 * np.sum(weights ** 2)
        return l1_loss + l2_loss

    def gradient(self, weights):
        """
        计算正则化梯度
        :param weights: 模型权重
        :return: 正则化梯度
        """
        l1_grad = self.l1 * np.sign(weights)
        l2_grad = 2 * self.l2 * weights
        return l1_grad + l2_grad

# 定义自定义GBDT模型
class GBDT:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, l1=0.0, l2=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.regularization = Regularization(l1=l1, l2=l2)

    def fit(self, X, y):
        residual = y.copy()  # 初始残差等于目标值
        for i in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residual)  # 用当前残差训练回归树
            predictions = tree.predict(X)  # 用回归树预测残差
            residual -= self.learning_rate * predictions  # 更新残差
            self.trees.append(tree)
            # 输出训练进度
            print(f"Tree {i + 1}/{self.n_estimators} trained with max depth {self.max_depth}")

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return (predictions > 0.5).astype(int)

# 模型训练与保存
def train_and_save_model(train_file, model_file):
    X_train, y_train = load_data(train_file)
    model = GBDT(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)

# 模型测试与评估
def test_and_evaluate_model(test_file, model_file, result_file, chart_file):
    X_test, y_test = load_data(test_file)
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    predictions = model.predict(X_test)

    normal_count = np.sum(predictions == 0)
    attack_count = np.sum(predictions == 1)
    correct_attack = np.sum((predictions == 1) & (y_test == 1))
    incorrect_attack = np.sum((predictions == 1) & (y_test == 0))

    accuracy = np.mean(predictions == y_test)
    precision = correct_attack / (correct_attack + incorrect_attack) if (correct_attack + incorrect_attack) > 0 else 0
    recall = correct_attack / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0

    with open(result_file, 'w') as file:
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")
        file.write(f"Normal Count: {normal_count}\n")
        file.write(f"Attack Count: {attack_count}\n")
        file.write(f"Correct Attack: {correct_attack}\n")
        file.write(f"Incorrect Attack: {incorrect_attack}\n")

    plt.bar(['Normal', 'Attack'], [normal_count, attack_count], color=['blue', 'red'])
    plt.title('Prediction Distribution')
    plt.savefig(chart_file)

# 主程序入口
if __name__ == "__main__":
    train_file = "./data/NSL-KDD-DataSet/KDDTrain+.csv"
    test_file = "./data/NSL-KDD-DataSet/KDDTest+.csv"
    model_file = "./gbdt_model.pkl"
    result_file = "./test_results.csv"
    chart_file = "./prediction_chart.png"

    train_and_save_model(train_file, model_file)
    test_and_evaluate_model(test_file, model_file, result_file, chart_file)
