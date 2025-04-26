# -*- coding:utf-8 -*-
from datetime import datetime
import abc
from random import sample
from math import exp, log
from gbdt.tree import construct_decision_tree
import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class RegressionLossFunction(metaclass=abc.ABCMeta):
    def __init__(self, n_classes):
        self.K = n_classes

    @abc.abstractmethod
    def compute_residual(self, dataset, subset, f):
        """计算残差"""

    @abc.abstractmethod
    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        """更新F_{m-1}的值"""

    @abc.abstractmethod
    def initialize(self, f, dataset):
        """初始化F_{0}的值"""

    @abc.abstractmethod
    def update_ternimal_regions(self, targets, idset):
        """更新叶子节点的返回值"""


class LeastSquaresError(RegressionLossFunction):
    """用于回归的最小平方误差损失函数"""
    def __init__(self, n_classes):
        if n_classes != 1:
            raise ValueError("``n_classes`` must be 1 for regression but "
                             "was %r" % n_classes)
        super(LeastSquaresError, self).__init__(n_classes)

    def compute_residual(self, dataset, subset, f):
        residual = {}
        for id in subset:
            y_i = dataset.get_instance(id)['label']
            residual[id] = y_i - f[id]
        return residual

    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        data_idset = set(dataset.get_instances_idset())
        subset = set(subset)
        for node in leaf_nodes:
            for id in node.get_idset():
                f[id] += learn_rate*node.get_predict_value()
        for id in data_idset-subset:
            f[id] += learn_rate*tree.get_predict_value(dataset.get_instance(id))

    def initialize(self, f, dataset):
        """初始化F0，我们可以用训练样本的所有值的平均值来初始化，为了方便，这里初始化为0.0"""
        ids = dataset.get_instances_idset()
        for id in ids:
            f[id] = 0.0

    def update_ternimal_regions(self, targets, idset):
        sum1 = sum([targets[id] for id in idset])
        return sum1/len(idset)


class ClassificationLossFunction(metaclass=abc.ABCMeta):
    """分类损失函数的基类"""
    def __init__(self, n_classes):
        self.K = n_classes

    @abc.abstractmethod
    def compute_residual(self, dataset, subset, f):
        """计算残差"""

    @abc.abstractmethod
    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        """更新F_{m-1}的值"""

    @abc.abstractmethod
    def initialize(self, f, dataset):
        """初始化F_{0}的值"""

    @abc.abstractmethod
    def update_ternimal_regions(self, targets, idset):
        """更新叶子节点的返回值"""


class BinomialDeviance(ClassificationLossFunction):
    """二元分类的损失函数"""
    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes.".format(
                self.__class__.__name__))
        super(BinomialDeviance, self).__init__(1)

    def compute_residual(self, dataset, subset, f):
        residual = {}
        for id in subset:
            y_i = dataset.get_instance(id)['label']
            residual[id] = 2.0*y_i/(1+exp(2*y_i*f[id]))
        return residual

    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        data_idset = set(dataset.get_instances_idset())
        subset = set(subset)
        for node in leaf_nodes:
            for id in node.get_idset():
                f[id] += learn_rate*node.get_predict_value()
        for id in data_idset-subset:
            f[id] += learn_rate*tree.get_predict_value(dataset.get_instance(id))

    def initialize(self, f, dataset):
        ids = dataset.get_instances_idset()
        for id in ids:
            f[id] = 0.0

    def update_ternimal_regions(self, targets, idset):
        sum1 = sum([targets[id] for id in idset])
        if sum1 == 0:
            return sum1
        sum2 = sum([abs(targets[id])*(2-abs(targets[id])) for id in idset])
        return sum1 / sum2


class MultinomialDeviance(ClassificationLossFunction):
    """多元分类的损失函数"""
    def __init__(self, n_classes, labelset):
        self.labelset = set([label for label in labelset])
        if n_classes < 3:
            raise ValueError("{0:s} requires more than 2 classes.".format(
                self.__class__.__name__))
        super(MultinomialDeviance, self).__init__(n_classes)

    def compute_residual(self, dataset, subset, f):
        label_valueset = dataset.get_label_valueset()
        residual = pd.DataFrame(index=subset, columns=label_valueset)
        for id in subset:
            p_sum = sum([exp(f[id][x]) for x in label_valueset])
            for label in label_valueset:
                p = exp(f[id][label]) / p_sum
                y = 1.0 if dataset.data.loc[id, "label"] == label else 0.0
                residual.loc[id, label] = y - p
        return residual

    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        data_idset = set(dataset.get_instances_idset())
        subset = set(subset)
        for node in leaf_nodes:
            for id in node.get_idset():
                f[id][label] += learn_rate*node.get_predict_value()
        # 更新OOB的样本
        for id in data_idset-subset:
            f[id][label] += learn_rate*tree.get_predict_value(dataset.get_instance(id))

    def initialize(self, f, dataset):
        ids = dataset.get_instances_idset()
        for id in ids:
            f[id] = dict()
            for label in dataset.get_label_valueset():
                f[id][label] = 0.0

    def update_ternimal_regions(self, targets, idset):
        sum1 = sum([targets[id] for id in idset])
        if sum1 == 0:
            return sum1
        sum2 = sum([abs(targets[id])*(1-abs(targets[id])) for id in idset])
        return ((self.K-1)/self.K)*(sum1/sum2)


class GBDTMultiClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []  # 每个类别的树
        self.classes_ = None

    def fit(self, X, y):
        # 获取类别标签
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # 初始化F值
        F = np.zeros((X.shape[0], n_classes))

        # One-hot编码标签
        y_one_hot = np.zeros((X.shape[0], n_classes))
        for i, label in enumerate(self.classes_):
            y_one_hot[:, i] = (y == label).astype(int)

        for estimator_idx in range(self.n_estimators):
            print(f"Starting training for tree {estimator_idx + 1}/{self.n_estimators}...")
            iteration_start_time = time.time()

            trees_for_iteration = []
            avg_residuals = []

            for class_idx in range(n_classes):
                # 计算残差
                residual = y_one_hot[:, class_idx] - self._softmax(F)[:, class_idx]
                avg_residual = np.mean(np.abs(residual))
                avg_residuals.append(avg_residual)

                # 拟合残差
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X, residual)
                trees_for_iteration.append(tree)

                # 更新F值
                F[:, class_idx] += self.learning_rate * tree.predict(X)

            self.trees.append(trees_for_iteration)

            iteration_elapsed_time = time.time() - iteration_start_time
            print(f"Tree {estimator_idx + 1} completed. Average residual: {np.mean(avg_residuals):.6f}, Training time: {iteration_elapsed_time:.2f} seconds.")

    def predict(self, X):
        # 初始化F值
        F = np.zeros((X.shape[0], len(self.classes_)))

        for trees_for_iteration in self.trees:
            for class_idx, tree in enumerate(trees_for_iteration):
                F[:, class_idx] += self.learning_rate * tree.predict(X)

        # 返回概率最大的类别
        return self.classes_[np.argmax(self._softmax(F), axis=1)]

    def _softmax(self, F):
        exp_F = np.exp(F - np.max(F, axis=1, keepdims=True))  # 防止溢出
        return exp_F / np.sum(exp_F, axis=1, keepdims=True)
