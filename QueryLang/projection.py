import random
import numpy as np
import pandas as pd
from docplex.mp.model import Model


def correct_sign(sign):
    if sign == '<':
        return '<='
    if sign == '>':
        return '>='
    if sign == '=':
        return '=='
    return sign


def _signs_flip():
    return {'==': '!=',
            '=': '!=',
            '<': '>=',
            '<=': ">=",
            '>': '<=',
            '>=': '<=',
            '!=': '==',
            }


class Project:
    def __init__(self, original, to_project, encoder, conjunctions, is_neural=False):
        self.integer_epsilon = 1
        self.numeric_epsilon = 0.01
        self.plus_infinity = 1e+20
        self.encoder = encoder
        self.original = original
        self.to_project = to_project
        self.model = Model()
        self.n_features = len(encoder.features)
        self.n_categorical = len(encoder.columns)
        self.init_model()
        self.signs_flip = _signs_flip()
        self.parse_conjunctions(conjunctions)
        self.solution = self.minimize(is_neural)

    def init_model(self):
        for i, feature in enumerate(self.encoder.features):
            if i < self.n_categorical:  # categorical features
                y_i = self.model.integer_var(lb=0, ub=self.encoder.cat_vars_ord[i] - 1, name=f'y_{feature}')
                C_i = self.model.binary_var(name=f'C_{feature}')
                self.model.add_if_then(y_i != self.original[i], C_i == 1, negate=False)
                self.model.add_if_then(y_i == self.original[i], C_i == 0, negate=False)

                C_i_p = self.model.binary_var(name=f'C_{feature}_p')
                self.model.add_if_then(y_i != int(self.to_project[i]), C_i_p == 1, negate=False)
                self.model.add_if_then(y_i == int(self.to_project[i]), C_i_p == 0, negate=False)

            else: # numeric features
                if self.is_integer(feature):
                    y_i = self.model.integer_var(lb=-self.plus_infinity,name=f'y_{feature}') # range [-inf,inf)
                    epsilon = self.integer_epsilon
                else:
                    y_i = self.model.continuous_var(lb=-self.plus_infinity, name=f'y_{feature}')
                    epsilon = self.numeric_epsilon
                C_i = self.model.binary_var(name=f'C_{feature}')
                C_i_p = self.model.binary_var(name=f'C_{feature}_p')

                self.model.add_indicator(C_i, self.model.abs(y_i - self.original[i]) >= epsilon, active_value=1,
                                         name=f'C_{i}==1,1')
                self.model.add_indicator(C_i, y_i == self.original[i], active_value=0, name=f'C_{i}==1,1')


                self.model.add_indicator(C_i_p, self.model.abs(y_i - self.to_project[i]) >= epsilon, active_value=1,
                                         name=f'C_{i}_p==1,1')
                self.model.add_indicator(C_i_p, y_i == self.to_project[i], active_value=0, name=f'C_{i}_p==1,1')

    def get_epsilon(self, feature):
        return self.integer_epsilon if self.is_integer(feature) else self.numeric_epsilon

    def is_integer(self, feature):
        if self.encoder.dtypes[feature] == np.int64 or self.encoder.dtypes[feature] == np.int32:
            return True
        return False

    def encode(self, feature, val):
        return self.encoder.label_encoder[feature][val]

    def parse_conjunctions(self, conjunctions):
        if len(conjunctions) > 0:
            constraints = [self.parse_atom(atom) for atom in conjunctions]
            self.model.add_constraints(constraints)

    def minimize(self, is_neural=False):

        if is_neural:
            objective = self.model.sum(self.model.get_var_by_name(name=f'C_{f}_p')
                                if i < self.n_categorical
                                else self.model.abs(self.model.get_var_by_name(name=f'y_{f}') - self.to_project[i])/self.encoder.mads[f]
                                       for i, f in enumerate(self.encoder.features))
        else:
            objective = self.model.sum(self.model.get_var_by_name(name=f'C_{f}_p')
                                if i < self.n_categorical
                                else self.model.abs(self.model.get_var_by_name(name=f'y_{f}') - self.to_project[i])/self.encoder.mads[f]
                                       for i, f in enumerate(self.encoder.features))
        self.model.minimize(objective)
        return self.model.solve()

    def is_feasible(self):
        return self.solution is not None

    def get_project(self, decode=False):
        if not self.is_feasible():
            return None
        projected_values = []
        for i, f in enumerate(self.encoder.features):
            val = self.solution.get_value(f'y_{f}')
            if i < self.n_categorical:
                val = int(val)
            projected_values.append(val)
        p = pd.DataFrame(data=[projected_values], columns=self.encoder.features)
        if not decode:
            return p
        return self.encoder.inverse_transform_to_project(p)

    def parse_atom(self, atom):
        sign = atom[0]
        negate = 1
        if sign == "~":
            atom = atom[1:]
            negate = 0
        if atom[0] != '(':  # C_feature == 0/1
            return eval('self.model.get_var_by_name(atom) == ' + str(negate))
        else:
            left, sign, right = atom[1:-1].split()
            sign = correct_sign(sign)
            if negate == 0:
                sign = self.signs_flip[sign]
            if left[0] == 'y' and right[0] == 'y':  # y_f1 op y_f2
                return eval('self.model.get_var_by_name(left)' + str(sign) + 'self.model.get_var_by_name(right)')
            else:  # y_f1 op val or val op y_f1
                if right[0] == 'y':
                    left, right = right, left
                    sign = self.signs_flip[sign]
                if left[2:] in self.encoder.columns:
                    right = right.strip("'\'")
                    return eval('self.model.get_var_by_name(left)' + str(sign) + 'self.encode(left[2:],right)')
                else:
                    if sign == '!=':
                        return eval('self.model.abs(self.model.get_var_by_name(left) - ' + str(right) + ' ) >= self.get_epsilon(left[2:])')
                    return eval('self.model.get_var_by_name(left)' + str(sign) + str((eval(right))))


def get_epsilon(encoder, feature):
    if encoder.dtypes[feature] == np.int64 or encoder.dtypes[feature] == np.int32:
        return 1
    return 0.01


def check_atom(encoder, atom, Cs, ys):
    sign = atom[0]
    negate = False
    if sign == "~":
        atom = atom[1:]
        negate = True
    if atom[0] != '(':  # C_feature == 0/1
        return not (atom in Cs) if negate else atom in Cs
    else:
        left, sign, right = atom[1:-1].split()
        sign = correct_sign(sign)
        if negate:
            sign = _signs_flip()[sign]
        if left[0] == 'y' and right[0] == 'y':  # y_f1 op y_f2
            return eval('ys[left]' + str(sign) + 'ys[right]')
        else:  # y_f1 op val or val op y_f1
            if right[0] == 'y':
                left, right = right, left
                sign = _signs_flip()[sign]
            if hasattr(encoder, 'categorical_feature_names'):
                categorical = encoder.categorical_feature_names
            else:
                categorical = encoder.columns
            if left[2:] in categorical:
                return eval('ys[left]' + str(sign) + 'right')
            else:
                if sign == '!=':
                    return eval('ys[left]' + '!=' + right)
                return eval('ys[left]' + str(sign) + str(right))


def get_all_projected(instance, to_project, encoder, dnf_expression, decode=False, k=3, is_neural=False):
    projects = []
    random.shuffle(dnf_expression)
    for c in dnf_expression:
        p = Project(instance, to_project, encoder, c, is_neural)
        if p.is_feasible():
            projects.append((p.get_project(decode=decode)))
        if len(projects) == min(k, len(dnf_expression)):
            break
    if len(projects) > 0:
        return pd.concat(projects)
    return None


def random_subset(s):
    out = set()
    while len(out) == 0:
        out = set()
        for el in s:
            # random coin flip
            if random.randint(0, 1) == 0:
                out.add(el)
    return out