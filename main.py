import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def read_data(file_path):
    """
    Reads the network traffic dataset and returns the data, labels, and label mapping.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        tuple: A tuple containing the data (DataFrame), labels (Series), and label mapping (dict).
    """
    # Read the dataset without headers
    data = pd.read_csv(file_path, header=None)  # Read the file without assuming headers

    # Encode categorical labels if necessary
    label_mapping = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category')
            if col == data.columns[-2]:  # Assuming the second to last column contains labels
                label_mapping = dict(enumerate(data[col].cat.categories))
            data[col] = data[col].cat.codes

    # Extract labels (second to last column)
    labels = data.iloc[:, -2]

    # Extract features (all columns except the last two)
    features = data.iloc[:, :-2]

    return features, labels, label_mapping

def train_gbdt(features, labels):
    """
    Trains a Gradient Boosting Decision Tree (GBDT) model using the provided features and labels.

    Args:
        features (DataFrame): The feature set for training.
        labels (Series): The corresponding labels for the feature set.

    Returns:
        GradientBoostingClassifier: The trained GBDT model.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize the GBDT model with custom parameters
    gbdt_model = GradientBoostingClassifier(
        n_estimators=100,       # Number of trees
        learning_rate=0.1,      # Learning rate
        criterion='friedman_mse',  # Use 'friedman_mse' as the criterion
        max_depth=3,            # Maximum depth of each tree
        random_state=42         # Random state for reproducibility
    )

    # Train the model
    gbdt_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = gbdt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"GBDT Model Accuracy: {accuracy:.2f}")

    return gbdt_model

def main():
    """
    Main function to read the traffic data and save it to a temporary file.
    """
    # Define file paths
    input_file = "data\\NSL-KDD-DataSet\\KDDTest+.csv"
    output_file = "./temp2.csv"
    model_file = "./gbdt_model.pkl"  # File to save the trained model

    features, labels, label_mapping = read_data(input_file)

    combined_data = pd.concat([features, labels.rename("Label")], axis=1)   
    combined_data.to_csv(output_file, index=False, header=False)

    print("Label Mapping:", label_mapping)
    gbdt_model = train_gbdt(features, labels)

    joblib.dump(gbdt_model, model_file)
    print(f"Trained GBDT Model saved to: {model_file}")

    print("Trained GBDT Model:", gbdt_model)


if __name__ == "__main__":
    main()