# -*- coding:utf-8 -*-
import pandas as pd

class DataSet:
    """
    分类问题默认标签列名称为label，二元分类标签∈{-1, +1}
    回归问题也统一使用label
    """
    def __init__(self, features: pd.DataFrame, labels: pd.Series):
        self.instances = dict()
        self.field_names = list(features.columns) + ["label"]
        self.field_type = {col: set() for col in features.columns}
        self.distinct_valueset = {col: set() for col in features.columns}

        for idx, row in features.iterrows():
            instance = row.to_dict()
            for col in features.columns:
                self.distinct_valueset[col].add(instance[col])
            instance["label"] = labels.loc[idx]
            self.instances[idx] = instance

    def get_instances_idset(self):
        """获取样本的id集合"""
        return set(self.instances.keys())

    def is_real_type_field(self, name):
        """判断特征类型是否是real type"""
        if name not in self.field_names:
            raise ValueError(" field name not in the dictionary of dataset")
        return len(self.field_type[name]) == 0

    def get_label_size(self, name="label"):
        if name not in self.field_names:
            raise ValueError(" there is no class label field!")
        return len(set(instance[name] for instance in self.instances.values()))

    def get_label_valueset(self, name="label"):
        """返回具体分离label"""
        if name not in self.field_names:
            raise ValueError(" there is no class label field!")
        return set(instance[name] for instance in self.instances.values())

    def size(self):
        """返回样本个数"""
        return len(self.instances)

    def get_instance(self, Id):
        """根据ID获取样本"""
        if Id not in self.instances:
            raise ValueError("Id not in the instances dict of dataset")
        return self.instances[Id]

    def get_attributes(self):
        """返回所有features的名称"""
        return [x for x in self.field_names if x != "label"]

    def get_distinct_valueset(self, name):
        if name not in self.field_names:
            raise ValueError("the field name not in the dataset field dictionary")
        return self.distinct_valueset[name] if name in self.distinct_valueset else set()

if __name__ == "__main__":
    from sys import argv
    import pandas as pd

    # 示例数据
    features = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    labels = pd.Series([0, 1, 0])

    data = DataSet(features, labels)
    print("instances size=", len(data.instances))
    print(data.instances)
