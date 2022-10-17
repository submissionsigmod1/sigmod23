import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from projection import get_all_projected


def get_prediction_label(tensor):
    tensor = torch.round(tensor)
    return tensor


class NeuralCec:
    def __init__(self, model, enc):
        self.epsilon = 0.15
        self.lr = 0.001
        self.max_iter = 25
        self.model = model
        self.enc = enc
        self.dnf_expression = None

    def get_cfs(self, instance, to_class, conditions):
        func, args = conditions
        self.dnf_expression = func(self.enc, instance, **args, is_dnf=True)
        to_class = torch.FloatTensor([[to_class]])
        pred = 1-to_class
        columns = instance.columns
        cf = torch.FloatTensor(self.enc.transform(instance).values)

        for i in range(self.max_iter):
            cf, data_grad = self.pertub(cf, to_class)  # tensor
            if abs(to_class-self.model(cf)).item() > 0.5:
                continue
            cf = self.fix_enc(cf, data_grad) #numpy
            cf = pd.DataFrame(data=[cf], columns=columns)  # pandas
            cf = self.enc.inverse_transform(cf)  # numpy
            cf = self.enc.transform_to_project(cf)  #numpy
            pd_cf = self.project(instance, cf).reset_index(drop=True)
            cf = torch.FloatTensor(self.enc.transform(pd_cf).values)
            logit = self.model(cf)
            if to_class.item() == 1:
                idx = torch.argmax(logit).item()
            else:
                idx = torch.argmin(logit).item()
            cf = cf[[idx]]
            logit = logit[[idx]]
            pred = get_prediction_label(logit)
            if pred == to_class:
                break
        return (pred == to_class).item(), pd_cf.loc[[idx]].reset_index(drop=True)

    def get_data_grad(self, cf, to_class):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        optimizer.zero_grad()
        self.model.zero_grad()
        cf = torch.detach(cf)
        cf.requires_grad = True
        cf_pred = self.model(cf)
        loss = criterion(cf_pred, to_class)
        loss.backward()
        return cf.grad.data

    def pertub(self, cf, to_class):
        for i in range(6*self.max_iter):
            data_grad = self.get_data_grad(cf, to_class)
            epsilon = random.randrange(1, 5)/10
            cf = cf - epsilon*data_grad

            cf = torch.detach(cf)
            for feature in range(len(cf[0])):
                if self.enc.is_category(feature) and cf[0][feature] < 0:
                    try:
                        cf[0][feature] = min(self.enc.encoder[self.enc.features[feature]].values())-0.01
                    except Exception:
                        cf[0][feature] = min(self.enc.encoder[self.enc.features[feature]])-0.01

            logit = self.model(cf)
            cf_pred = get_prediction_label(logit)

            data_grad = self.get_data_grad(cf, to_class)
            if cf_pred == to_class and abs(to_class-logit).item() < random.randrange(1, 5)/10:
                break
        return cf, data_grad

    def fix_enc(self, cf, data_grad):
        grad_sign = data_grad.sign().detach().numpy()[0]
        cf = cf.detach().numpy()[0]
        grad_sign = grad_sign.astype(np.float64)
        cf = cf.astype(np.float64)
        for feature in range(len(cf)):
            if self.enc.is_category(feature):
                if grad_sign[feature] == 1:
                    cf[feature] = self.enc.closest(feature, cf[feature], bigger=True)
                else:
                    cf[feature] = self.enc.closest(feature, cf[feature], bigger=False)
            else:
                if self.enc.dtypes[feature] == np.int64 or self.enc.dtypes[feature] == np.int32:
                    n = cf[feature]
                    if grad_sign[feature] == 1:
                        cf[feature] = int(n)+1
                    else:
                        if int(n) == n:
                            cf[feature] = n-1
                        else:
                            cf[feature] = int(n)
        return cf

    def project(self, instance, cf):
        instance_to_project = self.enc.transform_to_project(instance).values[0]
        p = get_all_projected(instance_to_project, cf.values[0], self.enc, self.dnf_expression, k=3, is_neural=True, decode=True)
        return p