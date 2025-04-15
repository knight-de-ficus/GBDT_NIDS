import csv
import numpy as np
from collections import Counter

class XGBoost:
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, reg_lambda=1.0, reg_alpha=0.0):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.reg_lambda = reg_lambda  # L2正则化系数
        self.reg_alpha = reg_alpha    # L1正则化系数
        self.trees = []

    def fit(self, X, y):
        # 初始化预测值为样本均值
        y_pred = np.full(y.shape, np.mean(y))
        for _ in range(self.n_estimators):
            residual = y - y_pred
            tree = self._build_tree(X, residual, depth=0)
            self.trees.append(tree)
            y_pred += self.learning_rate * self._predict_tree(tree, X)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(tree, X)
        return np.round(y_pred)

    def _build_tree(self, X, residual, depth):
        if depth >= self.max_depth or len(set(residual)) == 1:
            return np.mean(residual)
        best_split = self._find_best_split(X, residual)
        if not best_split:
            return np.mean(residual)
        left_idx = X[:, best_split['feature']] <= best_split['threshold']
        right_idx = ~left_idx
        left_tree = self._build_tree(X[left_idx], residual[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx], residual[right_idx], depth + 1)
        return {'feature': best_split['feature'], 'threshold': best_split['threshold'], 'left': left_tree, 'right': right_tree}

    def _find_best_split(self, X, residual):
        best_split = None
        best_loss = float('inf')
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                left_loss = np.var(residual[left_idx]) * np.sum(left_idx)
                right_loss = np.var(residual[right_idx]) * np.sum(right_idx)
                
                # 添加正则化项
                reg_term = self.reg_lambda * (np.sum(left_idx) + np.sum(right_idx)) + self.reg_alpha
                loss = left_loss + right_loss + reg_term
                
                if loss < best_loss:
                    best_loss = loss
                    best_split = {'feature': feature, 'threshold': threshold}
        return best_split

    def _predict_tree(self, tree, X):
        if not isinstance(tree, dict):
            return np.full(X.shape[0], tree)
        left_idx = X[:, tree['feature']] <= tree['threshold']
        right_idx = ~left_idx
        y_pred = np.zeros(X.shape[0])
        y_pred[left_idx] = self._predict_tree(tree['left'], X[left_idx])
        y_pred[right_idx] = self._predict_tree(tree['right'], X[right_idx])
        return y_pred

def load_data(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    X = np.array([list(map(float, row[:-2])) for row in data])
    y = np.array([1 if row[-2] != 'normal' else 0 for row in data])
    return X, y

def main():
    train_file = "./KDDTrain+.txt"
    test_file = "./KDDTest+.txt"

    # 加载数据
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)

    # 训练模型，添加正则化参数
    model = XGBoost(max_depth=5, learning_rate=0.1, n_estimators=50, reg_lambda=1.0, reg_alpha=0.1)
    model.fit(X_train, y_train)

    # 测试模型
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"测试集准确率: {accuracy:.2f}")

if __name__ == "__main__":
    main()
