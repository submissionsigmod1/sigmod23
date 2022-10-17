from operator import itemgetter

import math
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import numpy as np
from decision_tree import DecisionTree
import itertools
import random
import pickle
from projection import get_all_projected


class Cec:

    def __init__(self, clf):
        self.model = clf
        self.clf = clf.steps[1][1]
        self.encoder = clf.steps[0][1]
        self.trees = self.get_trees()
        self.max_iter = 10
        self.beta = 0.51
        self.epsilon = 0.01
        self.dnf_expression = None

    def l0(self, p1, p2):
        return sum((np.abs(np.array(p1) - np.array(p2))) != 0)

    def d_tabular(self, x, y):
        n = x.shape[1] if isinstance(x,pd.DataFrame) else x.shape[0]
        n_con = len(self.encoder.mads)
        n_cat = n - n_con
        d_con = 0
        d_cat = 0
        for i,f in enumerate(self.encoder.features):
            if isinstance(x,pd.DataFrame):
                v1, v2 = x[f].values[0], y[f].values[0]
            else: # numpy
                v1, v2 = x[i], y[i]
            if f not in self.encoder.columns:
                d_con += (abs(v1 - v2) / self.encoder.mads[f])
            else:
                if not v1 == v2:
                    d_cat += 1
        return (n_cat / n) * d_cat + (n_con / n) * d_con

    def l0_metric(self, x, y):
        x = x.to_numpy().reshape(-1)
        y = y.to_numpy().reshape(-1)
        return sum(x != y)

    def l1(self, p1, p2):
        return sum(np.abs(np.array(p1) - np.array(p2)))

    def get_trees(self):
        trees = []
        for tree in self.clf.estimators_:
            dt = DecisionTree(tree, self.encoder)
            trees.append(dt)
        return trees

    def optimized_algorithm(self, instance, actual_tag, constraints=lambda x: True):
        encoded_instance = self.encoder.transform(instance).values.squeeze()
        encoded_to_project = self.encoder.transform_to_project(instance).values[0]
        modified_instance = encoded_instance
        max_proba = 0
        func, args = constraints
        self.dnf_expression = func(self.encoder, instance, **args, is_dnf=True)
        for i in range(self.max_iter):
            modified_instances = []
            n_trees = len(self.trees)
            for decision_tree in random.sample(self.trees, k=min(max(int(math.sqrt(n_trees)),4),n_trees)):
                # perturb
                candidate_instances = decision_tree.modify_to_value(encoded_instance,
                                                                    modified_instance, actual_tag,
                                                                    d=self.d_tabular, k=3)
                # extend candidates
                if len(candidate_instances) > 0:
                    modified_instances.extend(candidate_instances)

            if len(modified_instances) == 0:
                break

            # project
            candidate_to_project = self.encoder.transform_to_project(self.encoder.inverse_transform(pd.DataFrame(data=modified_instances, columns=instance.columns))).values
            projs = [get_all_projected(encoded_to_project, candidate, self.encoder, self.dnf_expression, decode=False) for candidate in candidate_to_project]
            found = sum(x is not None for x in projs)
            if found == 0:
                break
            modified_df = pd.concat(projs)

            new_candidates = self.encoder.inverse_transform_to_project(modified_df)
            probas = self.model.predict_proba(new_candidates)

            probas = [p[actual_tag] for p in probas]
            if max(probas) < self.beta:
                max_index = np.argmax(probas)
            else:
                idx = np.where(np.array(probas) > self.beta)[0]
                f = lambda i: (i, self.d_tabular(instance, new_candidates.iloc[[i]]))
                max_index, _ = min(map(f, idx), key=itemgetter(1))
            modified_instance = self.encoder.transform(new_candidates.iloc[[max_index]]).values[0]
            max_proba = probas[max_index]

            if max_proba > self.beta:
                break

        modified_instance = self.encoder.inverse_transform(
            pd.DataFrame(data=[modified_instance], columns=instance.columns))
        return modified_instance, max_proba >= self.beta, i + 1

    def project(self, modified_instance, constraints):
        if constraints.__name__ == 'cec_q_dummy':
            # print(modified_instance)
            # modified_instance[0] = modified_instance[0]  +self.epsilon # manauly fix probelm with algorithm
            x1 = modified_instance[0]
            x2 = max(modified_instance[1], 0.5, modified_instance[0] + self.epsilon)
            # print(modified_instance)
            return [x1, x2]
        return modified_instance


def constraints(points_tuple):
    original = points_tuple[0]
    changed = points_tuple[1]

    number_changes = 0
    for o, c in zip(original, changed):
        if o != c:
            number_changes += 1

    return number_changes <= 5


def distance(point1, point2):
    return sum(np.abs(np.array(point1) - np.array(point2)))


