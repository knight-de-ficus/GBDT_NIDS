import pandas as pd
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


def preprocess_data(training_file_path, test_file_path, mode="one-hot"):
    """
    Reads the network traffic dataset and returns the data, labels, and label mapping.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        tuple: A tuple containing the data (DataFrame), labels (Series), and label mapping (dict).
    """
    
    training_data = pd.read_csv(training_file_path)
    test_data = pd.read_csv(test_file_path)

    training_labels = training_data.iloc[:, -2]
    training_features = training_data.iloc[:, :-2]
    test_labels = test_data.iloc[:, -2]
    test_features = test_data.iloc[:, :-2]

    # 新建一个字典 types，将 training_labels 中的元素映射到数字
    types = {label: idx for idx, label in enumerate(training_labels.unique())}
    print(types)

    # 对于字符串类型的列，根据 mode 参数选择编码方式
    for col in training_features.columns:
        if training_features[col].dtype == 'object':
            if mode == "huffman":
                huffman_dict, depth = huffman_encoding(training_features[col])
                # Map the column values to Huffman codes
                training_data_column = training_features[col].map(huffman_dict)
                test_data_column = test_features[col].map(huffman_dict)

                for i in range(depth):
                    training_features[f"{col}_huffman_{i}"] = training_data_column.apply(lambda x: x[i] if i < len(x) else '0')
                    test_features[f"{col}_huffman_{i}"] = test_data_column.apply(lambda x: str(x)[i] if isinstance(x, str) and i < len(x) else '0')
                # Drop the original column
                training_features.drop(columns=[col], inplace=True)
                test_features.drop(columns=[col], inplace=True)
                print(f"{depth}\n")
            elif mode == "one-hot":
                # Perform one-hot encoding
                one_hot_encoded = pd.get_dummies(training_features[col], prefix=col)
                training_features = pd.concat([training_features, one_hot_encoded], axis=1)
                training_features.drop(columns=[col], inplace=True)

                one_hot_encoded_test = pd.get_dummies(test_features[col], prefix=col)
                test_features = pd.concat([test_features, one_hot_encoded_test], axis=1)
                test_features.drop(columns=[col], inplace=True)

                # Align columns between training and test sets
                test_features = test_features.reindex(columns=training_features.columns, fill_value=0)

    # Create label_mapping and reverse_label_mapping
    label_mapping = {i: attack for i, attack in enumerate(types)}
    reverse_label_mapping = {attack: i for i, attack in label_mapping.items()}

    # Map training_labels and test_labels using reverse_label_mapping
    if training_labels.dtype == 'object':
        training_labels = training_labels.map(reverse_label_mapping)
        test_labels = test_labels.map(reverse_label_mapping)

    # Convert all feature column names to numeric values (column indices)
    training_features.columns = range(training_features.shape[1])
    test_features.columns = range(test_features.shape[1])

    return training_features, training_labels, test_features, test_labels, label_mapping
