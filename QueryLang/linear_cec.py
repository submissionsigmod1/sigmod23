import numpy as np
import pandas as pd

from projection import Project, get_all_projected


class LinearCec:
    def __init__(self, clf):
        self.epsilon = 0.01
        self.max_iter = 15
        self.intercept = clf[1].intercept_[0]
        self.coef = clf[1].coef_[0]
        self.c_2 = np.dot(self.coef, self.coef)
        self.clf = clf
        self.model = clf[1]
        self.enc = clf[0]
        self.dnf_expression = None

    def get_cfs(self, instance, to_class, conditions):
        func, args = conditions
        self.dnf_expression = func(self.enc, instance, **args, is_dnf=True)
        cf = instance
        columns = instance.columns
        for i in range(self.max_iter):
            cf = self.pertub(cf)
            cf = self.fix_enc(cf, to_class == 1)
            cf = pd.DataFrame(data=[cf], columns=columns)
            cf = self.enc.inverse_transform(cf)
            cf = self.enc.transform_to_project(cf)
            cf = self.project(instance, cf)
            pred = self.clf.predict(cf)[0]
            if pred == to_class:
                break
        return pred == to_class, cf

    def get_delta(self, x):
        signed_distance = np.dot(self.coef, x) + self.intercept
        delta = self.epsilon - signed_distance if signed_distance < 0 else -self.epsilon - signed_distance
        return delta / self.c_2

    def pertub(self, cf):
        x = self.enc.transform(cf)
        columns = x.columns
        x = [x[col].values[0] for col in columns]
        cf = np.zeros_like(x)
        for i in range(len(x)):
            cf[i] = x[i] + self.get_delta(x) * self.coef[i]
        return cf

    def fix_enc(self, cf, bigger):
        for feature in range(len(cf)):
            if self.enc.is_category(feature):
                if self.coef[feature] > 0:
                    if bigger:
                        cf[feature] = self.enc.closest(feature, cf[feature], bigger=True)
                    else:
                        cf[feature] = self.enc.closest(feature, cf[feature], bigger=False)
                else:
                    if bigger:
                        cf[feature] = self.enc.closest(feature, cf[feature], bigger=False)
                    else:
                        cf[feature] = self.enc.closest(feature, cf[feature], bigger=True)

            else:
                if self.enc.dtypes[feature] == np.int64 or self.enc.dtypes[feature] == np.int32:
                    n = cf[feature]
                    if self.coef[feature] > 0:
                        if bigger:
                            cf[feature] = int(n+1)
                        else:
                            cf[feature] = int(n-1)
                    else:
                        if bigger:
                            cf[feature] = int(n-1)
                        else:
                            cf[feature] = int(n+1)

        return cf

    def project(self, instance, cf):
        instance_to_project = self.enc.transform_to_project(instance).values[0]
        p = get_all_projected(instance_to_project, cf.values[0], self.enc, self.dnf_expression, decode=True)
        return p
