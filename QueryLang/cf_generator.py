from abc import ABC, abstractmethod
from dice_ml.explainer_interfaces.dice_pytorch import DicePyTorch
from typing import Union
import torch
import dice_ml
import pandas as pd
import numpy as np
# import logging
# import CFDB
import projection
from growingspheres import counterfactuals as gs
from linear_cec import LinearCec
from neural_cec import NeuralCec
from random_forest_cf_alg import Cec
from IPython.utils import io


def get_changes(instance, cfs:  Union[pd.DataFrame, None], outcome_name) -> Union[pd.DataFrame, None]:
    if cfs is None:
        return None

    if outcome_name in cfs:
        cfs.drop(columns=outcome_name, inplace=True)

    cfs = cfs.astype(instance.dtypes)
    cfs = cfs.reset_index(drop=True)

    duplicate_original = pd.concat([instance] * len(cfs), ignore_index=True)
    # duplicate_original
    ne_stacked = (duplicate_original != cfs).stack()
    changed = ne_stacked[ne_stacked]
    difference_locations = np.where(duplicate_original != cfs)
    changed_to = cfs.values[difference_locations]
    changes = pd.DataFrame({'NewValue': changed_to}, index=changed.index)
    changes.index.rename(['CfId', 'FeatureName'], inplace=True)
    # changes.reset_index(level=1, inplace=True)
    changes.reset_index(inplace=True)
    return changes


class CfGenerator(ABC):
    def __init__(self, classifier, classifier_id, encoder, model_type, outcome_name, cf_type):
        self.classifier = classifier
        self.classifier_id = classifier_id
        self.encoder = encoder
        self.model_type = model_type
        self.cf_type = cf_type
        self.outcome_name = outcome_name
        # self.params = params

    @abstractmethod
    def get_cfs(self, instance, to_class, conditions) -> Union[pd.DataFrame, None]:
        pass

    def get_changes(self, instance, cfs):
        return get_changes(instance, cfs, self.outcome_name)

class CecTreeCfGenerator(CfGenerator):
    def __init__(self, classifier, classifier_id, encoder, model_type, outcome_name, **params):
        super().__init__(classifier, classifier_id, encoder, model_type, outcome_name, 'CecCFs')
        self.cf_alg = Cec(classifier)
        self.features = params['features']
        self.feature_to_idx = {feature: idx for idx, feature in enumerate(self.features)}

    def get_cfs(self, instance, to_class, conditions) -> Union[pd.DataFrame, None]:
        optimized_instance, is_optimized_found, _ = self.cf_alg.optimized_algorithm(instance, to_class, conditions)
        cfs = optimized_instance if is_optimized_found else None
        return self.get_changes(instance, cfs)


class GrowingSpheresCfGenerator(CfGenerator):
    def __init__(self, classifier, classifier_id, encoder, model_type, outcome_name, **params):
        super().__init__(classifier, classifier_id, encoder, model_type, outcome_name, 'GrowingSpheresCFs')
        self.radius = 2.0
        if self.model_type in ['Neural_Cec', 'Neural_Dice']:
            self.predict = lambda x: torch.round(self.classifier(torch.FloatTensor(x))).detach().numpy().reshape(-1)
        else:
            self.predict = self.classifier[1].predict


    def get_cfs(self, instance, to_class, conditions) -> Union[pd.DataFrame, None]:
        func, args = conditions
        expression = func(self.encoder, instance.values[0], **args, is_dnf=False)
        gs_cf = gs.CounterfactualExplanation(instance, self.encoder, expression, self.outcome_name, self.predict, is_valid, method='GS')
        gs_cf.fit(n_in_layer=100, first_radius=self.radius, dicrease_radius=2.0, sparse=True, verbose=False)
        cfs = gs_cf.enemy
        if cfs is not None:
            cfs = cfs.reshape(1, -1)
            cfs = self.encoder.inverse_transform(pd.DataFrame(data=cfs, columns=self.encoder.features))
        return self.get_changes(instance, cfs)


class CecLinearCfGenerator(CfGenerator):
    def __init__(self, classifier, classifier_id, encoder, model_type, outcome_name, **params):
        super().__init__(classifier, classifier_id, encoder, model_type, outcome_name, 'CecCFs')
        self.cf_alg = LinearCec(classifier)
        self.features = params['features']
        self.feature_to_idx = {feature: idx for idx, feature in enumerate(self.features)}

    def get_cfs(self, instance, to_class, conditions) -> Union[pd.DataFrame, None]:
        is_found, cf = self.cf_alg.get_cfs(instance, to_class, conditions)
        cfs = cf if is_found else None
        return self.get_changes(instance, cfs)


class CecNeuralCfGenerator(CfGenerator):
    def __init__(self, classifier, classifier_id, encoder, model_type, outcome_name, **params):
        super().__init__(classifier, classifier_id, encoder, model_type, outcome_name, 'CecCFs')
        self.cf_alg = NeuralCec(classifier, encoder)
        self.features = params['features']
        self.feature_to_idx = {feature: idx for idx, feature in enumerate(self.features)}

    def get_cfs(self, instance, to_class, conditions) -> Union[pd.DataFrame, None]:
        is_found, cf = self.cf_alg.get_cfs(instance, to_class, conditions)
        cfs = cf if is_found else None
        return self.get_changes(instance, cfs)


class DiceCfGenerator(CfGenerator):

    def __init__(self, classifier, classifier_id, encoder, model_type, outcome_name, **params):
        super().__init__(classifier, classifier_id, encoder, model_type, outcome_name, 'DiverseCFs')
        d = dice_ml.Data(dataframe=params['dataframe'],
                         continuous_features=params['continuous_features'],
                         outcome_name=self.outcome_name)
        m = dice_ml.Model(model=classifier, backend=params['backend'])
        # Using method=random for generating CFs
        self.cf_generator = dice_ml.Dice(d, m, method=params['method'])
        self.total_CFs = params['total_CFs']
        self.columns = self.cf_generator.data_interface.data_df.columns.tolist()
        self.columns.remove(self.outcome_name)

    def get_cfs(self, instance, to_class, conditions):
        func, args = conditions
        func_name = func.__name__
        dice_input = instance.copy()
        if isinstance(self.cf_generator, DicePyTorch):
            dice_input = dice_input.to_dict(orient='records')[0]
        with io.capture_output():
            if func_name == "No_Con":
                cfs = self.cf_generator.generate_counterfactuals(dice_input, total_CFs=self.total_CFs,
                                                                 desired_class=to_class,
                                                                 verbose=False
                                                                 ).cf_examples_list[0].final_cfs_df
                cfs = self.is_correct_label(cfs, to_class)
                
            elif func_name == 'AND_NOT':
                features = set(args['features'])
                cols = set(self.columns.copy())
                to_vary = list(cols - features)
                cfs = self.cf_generator.generate_counterfactuals(dice_input, total_CFs=self.total_CFs,
                                                                 features_to_vary=to_vary,
                                                                 desired_class=to_class,
                                                                 verbose=False
                                                                 ).cf_examples_list[0].final_cfs_df
                cfs = self.is_correct_label(cfs, to_class)
                

            elif func_name == 'Q7':
                f_to_increase = args['f1']
                f_not_change = args['f2']
                to_vary = self.columns.copy()
                to_vary.remove(f_not_change)
                min_val = instance[f_to_increase].values[0]+1
                max_val = self.cf_generator.data_interface.get_features_range()[0][f_to_increase][1]+2
                permitted_range = {str(f_to_increase): [min_val, max_val]}
                cfs = self.cf_generator.generate_counterfactuals(dice_input, total_CFs=self.total_CFs,
                                                                 desired_class=to_class,
                                                                 features_to_vary=to_vary,
                                                                 permitted_range=permitted_range,
                                                                 verbose=False
                                                                 ).cf_examples_list[0].final_cfs_df

            elif func_name == 'Q9':
                f = args['f1']
                min_val = args['min_val']
                max_val = args['max_val']
                permitted_range = {str(f): [min_val, max_val]}
                cfs = self.cf_generator.generate_counterfactuals(dice_input, total_CFs=self.total_CFs,
                                                                 desired_class=to_class,
                                                                 permitted_range=permitted_range,
                                                                 verbose=False
                                                                 ).cf_examples_list[0].final_cfs_df

            elif func_name == 'Q10':
                f = args['f1']
                val = args['val']
                permitted_range = {str(f): [val]}
                cfs = self.cf_generator.generate_counterfactuals(dice_input, total_CFs=self.total_CFs,
                                                                 desired_class=to_class,
                                                                 permitted_range=permitted_range,
                                                                 verbose=False
                                                                 ).cf_examples_list[0].final_cfs_df
            else:
                print('add more queries')
                return None
        if cfs is None or len(cfs) == 0:
            return None
        full_cfs = cfs
        # changes = self.get_changes(instance, cfs)
        encoder = self.encoder
        expression = func(encoder, instance.values[0], **args, is_dnf=False)
        return get_valid_cfs(encoder, instance, full_cfs, expression, self.outcome_name)

    def is_correct_label(self, cfs, to_class):
        return cfs


def get_valid_cfs(encoder, instance, full_cfs, expression, outcome_name, return_only_changes=True):
    full_cfs = full_cfs.reset_index(drop=True)
    changes = get_changes(instance, full_cfs, outcome_name)
    valid_cfs = []
    n_valid = 0
    new_cf_ids = {}
    for idx in full_cfs.index:
        #Cs = set('C_' + changes[changes.CfId == idx].FeatureName)
        Cs = set(changes[changes.CfId == idx].FeatureName)
        for f in instance.columns.tolist():
            exec('C_'+f + f'=1 if f in Cs else 0')
        candidate = full_cfs.iloc[[idx]]
        ys = candidate.add_prefix('y_').to_dict(orient='records')[0]
        for key, val in ys.items():
            exec(key + '=val')
        if eval(expression):
        #if all([projection.check_atom(encoder, atom, Cs, ys) for atom in con]):
            if return_only_changes:
                valid = changes[changes.CfId == idx]
                new_cf_ids[idx] = n_valid
                valid_cfs.append(valid)
            else:
                valid_cfs.append(candidate)
            n_valid += 1
    if n_valid == 0:
        return None
    cfs = pd.concat(valid_cfs, ignore_index=True)
    if return_only_changes:
        return cfs.replace({"CfId": new_cf_ids})
    return encoder.transform(cfs).values


def is_valid(x, cf, enc, expression, outcome_name):
    x_df = enc.inverse_transform(pd.DataFrame(data=cf, columns=enc.features))
    instance = enc.inverse_transform(pd.DataFrame(data=x, columns=enc.features))
    return get_valid_cfs(enc, instance, x_df, expression, outcome_name, return_only_changes=False)

