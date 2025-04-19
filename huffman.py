import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from collections import Counter
import heapq

def huffman_encoding(data):
    """
    Perform Huffman encoding on a list of values.

    Args:
        data (pd.Series): Series of values to encode.

    Returns:
        tuple: A tuple containing:
            - pd.Series: Encoded column with Huffman codes.
            - int: Depth of the Huffman tree.
    """
    # Count the frequency of each value
    frequency = Counter(data)
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # Generate Huffman codes
    huffman_dict = {}
    for item in heap:
        for symbol, code in item[1:]:
            huffman_dict[symbol] = code

    # Calculate the depth of the Huffman tree
    max_depth = max(len(code) for code in huffman_dict.values())

    return huffman_dict, max_depth

def read_data(training_file_path, test_file_path):
    """
    Reads the network traffic dataset and returns the data, labels, and label mapping.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        tuple: A tuple containing the data (DataFrame), labels (Series), and label mapping (dict).
    """
    # Read the dataset without headers
    training_data = pd.read_csv(training_file_path, header=None)
    test_data = pd.read_csv(test_file_path, header=None)

    training_labels = training_data.iloc[:, -2]
    training_features = training_data.iloc[:, :-2]
    test_labels = test_data.iloc[:, -2]
    test_features = test_data.iloc[:, :-2]

    for col in training_features.columns:
        if training_features[col].dtype == 'object':
            huffman_dict, depth = huffman_encoding(training_features[col])
            # Map the column values to Huffman codes
            training_data_column = training_features[col].map(huffman_dict)
            test_data_column = test_features[col].map(huffman_dict)

            for i in range(depth):
                training_features[f"{col}_huffman_{i}"] = training_data_column.apply(lambda x: x[i] if i < len(x) else '0')
                test_features[f"{col}_huffman_{i}"] = test_data_column.apply(lambda x: x[i] if i < len(x) else '0')
            # Drop the original column
            training_features.drop(columns=[col], inplace=True)
            test_features.drop(columns=[col], inplace=True)
            print(f"{depth}\n")

    # Load attack types from attack_types.txt
    with open("data\\NSL-KDD-DataSet\\attack_types.txt", "r") as f:
        attack_types = [line.strip() for line in f.readlines()]

    # Create label_mapping and reverse_label_mapping
    label_mapping = {i: attack for i, attack in enumerate(attack_types)}
    reverse_label_mapping = {attack: i for i, attack in label_mapping.items()}

    # Map training_labels and test_labels using reverse_label_mapping
    if training_labels.dtype == 'object':
        training_labels = training_labels.map(reverse_label_mapping)
        test_labels = test_labels.map(reverse_label_mapping)

    return training_features, training_labels, test_features, test_labels, label_mapping, training_data, test_data

def train_gbdt(training_features, training_labels):
    """
    Train a Gradient Boosting Decision Tree (GBDT) model.

    Args:
        training_features (DataFrame): The feature set for training.
        training_labels (Series): The corresponding labels for the feature set.

    Returns:
        GradientBoostingClassifier: The trained GBDT model.
    """
    # Initialize the GBDT model
    gbdt_model = GradientBoostingClassifier(
        n_estimators=100,       # Number of trees
        learning_rate=0.1,      # Learning rate
        max_depth=3,            # Maximum depth of each tree
        random_state=42         # Random state for reproducibility
    )

    # Train the model
    gbdt_model.fit(training_features, training_labels)

    return gbdt_model

def train_xgboost(training_features, training_labels):
    """
    Train an XGBoost model.

    Args:
        training_features (DataFrame): The feature set for training.
        training_labels (Series): The corresponding labels for the feature set.

    Returns:
        xgb.Booster: The trained XGBoost model.
    """
    # Convert data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(training_features, label=training_labels)

    # Define XGBoost parameters
    params = {
        "objective": "multi:softmax",  # Multi-class classification
        "num_class": len(set(training_labels)),  # Number of classes
        "max_depth": 3,  # Maximum depth of a tree
        "eta": 0.1,  # Learning rate
        "seed": 42  # Random seed
    }

    # Train the model
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)

    return xgb_model

def main():
    """
    Main function to read the traffic data, train the model, and save results.
    """

    training_file_path = "data\\NSL-KDD-DataSet\\KDDTrain+.csv"
    test_file_path = "data\\NSL-KDD-DataSet\\KDDTest+.csv"

    training_features, training_labels, test_features, test_labels, label_mapping, training_data, test_data = read_data(training_file_path, test_file_path)

    print("Label Mapping:", label_mapping)

    training_features.columns = training_features.columns.astype(str)
    test_features.columns = test_features.columns.astype(str)

    # Train the XGBoost model
    xgb_model = train_xgboost(training_features, training_labels)

    # Predict using the XGBoost model
    dtest = xgb.DMatrix(test_features)
    test_predictions = xgb_model.predict(dtest)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"XGBoost Model Accuracy: {accuracy:.4f}")

    # Save results to result.csv
    result_df = pd.DataFrame({
        "True Label": test_labels,
        "Predicted Label": test_predictions,
        "Match": (test_labels == test_predictions).astype(int)
    })
    result_df.to_csv("./result.csv", index=False)
    print("Results saved to './result.csv'")

    # Save the trained model
    xgb_model.save_model("./xgb_model.json")
    print("Trained XGBoost model saved to './xgb_model.json'")

    training_features.to_csv("./temp2.csv", index=False, header=False)
    test_features.to_csv("./temp3.csv", index=False, header=False)

if __name__ == "__main__":
    main()