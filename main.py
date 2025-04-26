import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

from preprocess import preprocess_data
from gbdt.data import DataSet
from gbdt.model import GBDT
from gbdt.gbdt_model import GBDTMultiClassifier

def main():
    
    training_file_path = "data/UNSW_NB15_training-set.csv"
    test_file_path = "data/UNSW_NB15_testing-set.csv"

    training_features, training_labels, test_features, test_labels, label_mapping = preprocess_data(training_file_path, test_file_path, mode="one-hot")

    # # 将 training_features 和 training_labels 输出到 ./test.csv
    # training_data = pd.concat([training_features, training_labels], axis=1)
    # training_data.to_csv("./test.csv", index=False)
    # print("Training features and labels saved to './test.csv'")

    print("Label Mapping:", label_mapping)

    # 使用自定义的GBDT多分类模型
    gbdt_model = GBDTMultiClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    print("GBDTMultiClassifier initialized.")

    # 训练模型
    gbdt_model.fit(training_features.values, training_labels.values)
    print("GBDTMultiClassifier training completed.")

    # 保存模型到文件
    joblib.dump(gbdt_model, "./gbdt_model_500.pkl")
    print("Trained GBDTMultiClassifier model saved.")

    # 测试模型
    test_predictions = gbdt_model.predict(test_features.values)

    # 计算准确率
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"GBDTMultiClassifier Model Accuracy: {accuracy:.4f}")

    # 保存结果到CSV文件
    result_df = pd.DataFrame({
        "True Label": test_labels,
        "Predicted Label": test_predictions,
        "Match": (test_labels == test_predictions).astype(int)
    })
    result_df.to_csv("./result.csv", index=False)
    print("Results saved to './result.csv'")



if __name__ == "__main__":
    main()