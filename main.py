import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

from preprocess import preprocess_data
# from train_model import train_gbdt
from gbdt.data import DataSet
from gbdt.model import GBDT

def main():
    
    training_file_path = "data\\UNSW_NB15_testing-set.csv"
    test_file_path = "data\\UNSW_NB15_testing-set.csv"

    training_features, training_labels, test_features, test_labels, label_mapping = preprocess_data(training_file_path, test_file_path, mode="huffman")

    print("Label Mapping:", label_mapping)

    training_labels.to_csv("./temp2.csv", index=False, header=False)
    training_features.to_csv("./temp3.csv", index=False, header=False)

    gbdt_model = GBDT(max_iter=1, sample_rate=0.8, learn_rate=0.1, max_depth=3, loss_type='multi-classification')
    print("GBDT init ok.")

    dataset = DataSet(training_features, training_labels)
    print("Dataset init ok.")

    gbdt_model.fit(dataset, dataset.get_instances_idset())
    print("GBDT fit ok.")

    # test_predictions = [gbdt_model.predict_label(instance) for instance in test_features.to_dict(orient='records')]

    # accuracy = accuracy_score(test_labels, test_predictions)
    # print(f"GBDT Model Accuracy: {accuracy:.4f}")

    # result_df = pd.DataFrame({
    #     "True Label": test_labels,
    #     "Predicted Label": test_predictions,
    #     "Match": (test_labels == test_predictions).astype(int)
    # })
    # result_df.to_csv("./result.csv", index=False)
    # print("Results saved to './result.csv'")

    # joblib.dump(gbdt_model, "./gbdt_model.pkl")
    # print("Trained GBDT model saved to './gbdt_model.pkl'")


if __name__ == "__main__":
    main()