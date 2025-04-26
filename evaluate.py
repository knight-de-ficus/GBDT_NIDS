import pandas as pd

# Load attack types
# with open("data\\NSL-KDD-DataSet\\attack_types.txt", "r") as f:
#     attack_types = [line.strip() for line in f.readlines()]

# Correct label_mapping and reverse_label_mapping
label_mapping = {
    0: 'Normal', 1: 'Backdoor', 2: 'Analysis', 3: 'Fuzzers', 
    4: 'Shel', 6: 'Exploits', 7: 'DoS', 8: 'Worms', 9: 'Generic'
}
reverse_label_mapping = {attack: i for i, attack in label_mapping.items()}

# Load result.csv
result_df = pd.read_csv("./result.csv")

# Convert numeric labels to attack types
result_df["True Label"] = result_df["True Label"].map(label_mapping)
result_df["Predicted Label"] = result_df["Predicted Label"].map(label_mapping)

# Calculate accuracy
accuracy = (result_df["True Label"] == result_df["Predicted Label"]).mean()
print(f"Accuracy: {accuracy:.4f}")

# Calculate precision
true_positive = result_df[result_df["True Label"] == result_df["Predicted Label"]].groupby("Predicted Label").size()
predicted_positive = result_df.groupby("Predicted Label").size()
precision = (true_positive / predicted_positive).fillna(0)

# Calculate recall for each attack type
actual_positive = result_df.groupby("True Label").size()
recall = (true_positive / actual_positive).fillna(0)

# Display results
print("\nPrecision by Attack Type:")
print(precision)

print("\nRecall by Attack Type:")
print(recall)

