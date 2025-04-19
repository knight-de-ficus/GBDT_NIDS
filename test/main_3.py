import pandas as pd
import numpy as np
import pickle

class DataPreprocessor:
    def __init__(self, attack_types_path):
        self.attack_mapping = self._load_attack_types(attack_types_path)

    def _load_attack_types(self, path):
        with open(path, 'r') as f:
            attack_types = [line.strip() for line in f if line.strip()]
        return {attack: i for i, attack in enumerate(attack_types)}

    def preprocess(self, data_path):
        data = pd.read_csv(data_path, header=None)

        # Map categorical features (string) to integers
        for col in [1, 2, 3]:  # Protocol, service, flag
            if data[col].dtype == 'object':
                data[col] = data[col].astype('category').cat.codes

        # Map attack labels to indices based on attack_types
        data[41] = data[41].map(self.attack_mapping)

        # Drop the extra column (42) if it exists
        if 42 in data.columns:
            data = data.drop(columns=[42])

        X = data.iloc[:, :41].values
        y = data.iloc[:, 41].values
        return X, y

class DecisionTree:
    def __init__(self, max_depth=3, reg_lambda=1.0):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.tree = None

    def fit(self, X, y, residuals):
        self.tree = self._build_tree(X, residuals, depth=0)

    def _build_tree(self, X, residuals, depth):
        # Fix residuals handling to avoid unhashable type error
        if depth >= self.max_depth or len(np.unique(residuals)) == 1:
            # Add L2 regularization to the leaf value
            return np.mean(residuals) / (1 + self.reg_lambda)

        best_feature, best_threshold, best_loss = None, None, float('inf')
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                left_residuals = residuals[left_mask]
                right_residuals = residuals[right_mask]

                # Ensure no empty slices when calculating loss
                if len(left_residuals) == 0 or len(right_residuals) == 0:
                    return np.mean(residuals) / (1 + self.reg_lambda)

                # Calculate mean squared error (MSE) loss
                left_loss = np.mean((left_residuals - np.mean(left_residuals)) ** 2) if len(left_residuals) > 0 else 0
                right_loss = np.mean((right_residuals - np.mean(right_residuals)) ** 2) if len(right_residuals) > 0 else 0
                total_loss = left_loss * len(left_residuals) + right_loss * len(right_residuals)

                if total_loss < best_loss:
                    best_feature, best_threshold, best_loss = feature, threshold, total_loss

        if best_feature is None:
            return np.mean(residuals) / (1 + self.reg_lambda)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_tree = self._build_tree(X[left_mask], residuals[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], residuals[right_mask], depth + 1)

        return {"feature": best_feature, "threshold": best_threshold, "left": left_tree, "right": right_tree}

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        if not isinstance(tree, dict):  # 如果是叶子节点
            return tree
        if x[tree["feature"]] <= tree["threshold"]:  # 判断样本值是否满足分裂条件
            return self._predict_single(x, tree["left"])  # 递归进入左子树
        else:
            return self._predict_single(x, tree["right"])  # 递归进入右子树

class GBDT:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1.0, loss_function="mse", attack_mapping=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.loss_function = loss_function  # "mse" or "logloss"
        self.attack_mapping = attack_mapping  # Store attack_mapping
        self.trees = []

    def fit(self, X, y):
        # Correct label_to_index to assign continuous indices starting from 0
        label_to_index = {label: i for i, label in enumerate(sorted(set(self.attack_mapping.values()) | {'normal'}))}
        y_numeric = np.array([label_to_index.get(label, -1) for label in y])  # Convert y to numeric indices
        if -1 in y_numeric:
            raise ValueError("Some labels in y are not found in the attack mapping. Check your data or attack mapping.")

        # Ensure all attack types in y are present in attack_mapping
        unique_labels = set(y)
        missing_labels = unique_labels - set(label_to_index.keys())
        if missing_labels:
            raise ValueError(f"The following labels are missing in the attack mapping: {missing_labels}")

        predictions = np.zeros((y_numeric.shape[0], len(label_to_index)))  # One-hot encoded predictions
        for i in range(self.n_estimators):
            # Limit the number of trees to avoid overflow errors
            if i >= 100000:
                print("Stopping early to avoid overflow errors.")
                break
            residuals = self._compute_residuals(y_numeric, predictions)
            tree = DecisionTree(max_depth=self.max_depth, reg_lambda=self.reg_lambda)
            tree.fit(X, y_numeric, residuals)
            tree_predictions = tree.predict(X)
            # Ensure tree predictions match the shape of predictions
            tree_predictions_one_hot = np.zeros_like(predictions)
            for i, pred in enumerate(tree_predictions):
                tree_predictions_one_hot[i, int(pred)] = 1
            predictions += self.learning_rate * tree_predictions_one_hot
            self.trees.append(tree)
            avg_residual = np.mean(np.abs(residuals))
            print(f"Tree {i + 1}: Average residual = {avg_residual}")

    def _compute_residuals(self, y, predictions):
        if self.loss_function == "mse":
            return y - predictions
        elif self.loss_function == "logloss":
            # Gradient of logloss: y - softmax(predictions)
            exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))  # Numerical stability
            softmax = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

            # Convert y to one-hot encoding
            y_one_hot = np.zeros_like(predictions)
            if np.any(y >= predictions.shape[1]):  # 检查索引是否超出范围
                raise IndexError(f"Index in y is out of bounds for predictions. Max index: {predictions.shape[1] - 1}, y: {y}")
            y_one_hot[np.arange(y.shape[0]), y] = 1  # Set the correct class index to 1

            return y_one_hot - softmax
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.attack_mapping) + 1))  # One-hot encoding for all classes
        for tree in self.trees:
            tree_predictions = tree.predict(X)
            # Ensure tree predictions match the shape of predictions during prediction
            tree_predictions_one_hot = np.zeros_like(predictions)
            for i, pred in enumerate(tree_predictions):
                tree_predictions_one_hot[i, int(pred)] = 1
            predictions += self.learning_rate * tree_predictions_one_hot

        # For multi-class classification, take the argmax of the predictions
        class_indices = np.argmax(predictions, axis=1)

        # Map indices back to class labels (normal or attack types)
        index_to_label = {i: label for i, label in enumerate(['normal'] + list(self.attack_mapping.values()))}
        return np.array([index_to_label[idx] for idx in class_indices])

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

def evaluate_model(y_true, y_pred, average='macro'):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    tp = {}
    fp = {}
    fn = {}
    
    for cls in classes:
        tp[cls] = np.sum((y_true == cls) & (y_pred == cls))
        fp[cls] = np.sum((y_true != cls) & (y_pred == cls))
        fn[cls] = np.sum((y_true == cls) & (y_pred != cls))
    
    # 处理不同平均方式
    if average == 'macro':
        precision = np.mean([tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) != 0 else 0 for cls in classes])
        recall = np.mean([tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) != 0 else 0 for cls in classes])
    elif average == 'weighted':
        support = np.array([np.sum(y_true == cls) for cls in classes])
        precision = np.sum([(tp[cls] / (tp[cls] + fp[cls])) * support[cls] if (tp[cls] + fp[cls]) != 0 else 0 for cls in classes]) / np.sum(support)
        recall = np.sum([(tp[cls] / (tp[cls] + fn[cls])) * support[cls] if (tp[cls] + fn[cls]) != 0 else 0 for cls in classes]) / np.sum(support)
    else:  # micro
        total_tp = np.sum(list(tp.values()))
        total_fp = np.sum(list(fp.values()))
        total_fn = np.sum(list(fn.values()))
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else 0
    
    accuracy = np.mean(y_true == y_pred)
    return accuracy, precision, recall

def main():
    # Paths
    train_path = './data/NSL-KDD-DataSet/KDDTrain+.csv'
    test_path = './data/NSL-KDD-DataSet/KDDTest+.csv'
    attack_types_path = './data/NSL-KDD-DataSet/attack_types.txt'
    model_path = './gbdt_model.pkl'
    predictions_path = './predictions.csv'

    # Data preprocessing
    preprocessor = DataPreprocessor(attack_types_path)
    X_train, y_train = preprocessor.preprocess(train_path)
    X_test, y_test = preprocessor.preprocess(test_path)

    # # Train GBDT model
    # gbdt = GBDT(n_estimators=200, learning_rate=0.05, max_depth=4, reg_lambda=1.0, loss_function="logloss", attack_mapping=preprocessor.attack_mapping)
    # gbdt.fit(X_train, y_train)

    # # Save model
    # gbdt.save_model(model_path)

    # Load the pre-trained model
    gbdt = GBDT.load_model(model_path)

    # Predict and evaluate using the loaded model
    y_pred = gbdt.predict(X_test)
    accuracy, precision, recall = evaluate_model(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Save predictions
    pd.DataFrame({'True': y_test, 'Predicted': y_pred}).to_csv(predictions_path, index=False)

if __name__ == "__main__":
    main()
