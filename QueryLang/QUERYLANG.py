
from concurrent.futures import ProcessPoolExecutor
import sqlite3
import torch
from typing import Union
from itertools import product

import pandas as pd
import numpy as np
import collections

import sqlparse
from tqdm.notebook import tqdm


import cf_generator
import querylang_parser
from queries import get_prediction_query
import time
import os

import queries

PredictionsExample = collections.namedtuple("PredictionsExample", ["prediction_id", "instance_id", "z3_expression", "constraints", "label", "view_id"])
MAX_WORKERS = min(os.cpu_count(), 96)


class Instances(object):
    """ Class that represent instances of interest.

    Args:
      data: pandas dataframe.
      labels: series.
      favorite_class: desirable label.
    """
    def __init__(self, data: pd.DataFrame, labels: pd.Series, favorite_class):
        self.data = data.reset_index(drop=True)  # shape = (self.number_of_instances*len(self.feature_names) )
        self.labels = labels.reset_index(drop=True)  # shape 1*self.number_of_instance
        self.data.index.names = ['InstanceId']
        self.instances_relation = pd.merge(self.data, self.labels, left_index=True, right_index=True)
        self.outcome_name = labels.name
        self.feature_names = data.columns.tolist()
        self.classes = [0, 1]
        self.favorite_class = favorite_class
        self.continuous_features = self.get_continuous_features()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        return self.data.loc[[idx]], self.labels.loc[idx]

    def get_instance(self, idx):
        return self.__getitem__(idx)[0]

    def get_label(self, idx):
        return self.__getitem__(idx)[1]

    def get_continuous_features(self):
        g = self.data.columns.to_series().groupby(self.data.dtypes).groups
        if np.dtype('O') in g:
            return self.data.columns.difference(g[np.dtype('O')]).tolist()
        return self.data.columns.tolist()


def highlight_diff(cfs, original_instances, color='orange'):
    attr = 'background-color: {}'.format(color)
    return pd.DataFrame(np.where(cfs.ne(original_instances), attr, ''),
                        index=cfs.index, columns=cfs.columns)


def is_valid_view(cf_type, func_name):
    if cf_type == 'DiverseCFs' and func_name not in ['No_Con',
                                                     'AND_NOT',
                                                     'Q7',
                                                     'Q9',
                                                     'Q10']:
        return False
    return True


class QueryLang:
    """ Class that represent our propose QueryLang framework.

    Args:
      instances: Instances object.
      models: list of trained models and the encoding function that applied on the dataframe.
      db_name: database name.
      processes: number of CPU to use in the parallel mode
    """
    # assume dataframe,model have predict_prob function
    def __init__(self, instances: Instances, models, db_name=None, processes=2):
        # init database
        self.processes = processes
        if db_name is not None:
            self.db = sqlite3.connect(db_name)
        else:
            self.db = sqlite3.connect(':memory:')
        self.cur = self.db.cursor()

        # type Instances
        self.instances = instances

        self.models = models

        self.CF_generations = collections.defaultdict(dict)

        self.init_cf_generation()

        self.init_schema()
        self.init_instances_table()
        self.init_predictions_table()


        # suffix for CFs and Prediction_CFs relations: for example my_cfs_1, my_prediction_cfs_1
        self.suffix = 0
        self.cf_views = {}

    def init_schema(self):
        """initial the database"""

        create_classifiers = '''CREATE TABLE Classifiers (ClassifierId INTEGER, PRIMARY KEY(ClassifierId) );'''

        create_predictions = '''CREATE TABLE Predictions (PredictionId INTEGER PRIMARY KEY AUTOINCREMENT ,
                                      ClassifierId INTEGER ,
                                      InstanceId INTEGER ,
                                      Label INTEGER,
                                      Score REAL,
                                      
                                      
                                      FOREIGN KEY(ClassifierId) REFERENCES Classifiers(ClassifierId),
                                      FOREIGN KEY(InstanceId) REFERENCES Instances(InstanceId));'''


        create_prediction_cfs = '''CREATE TABLE Prediction_CFs (CfId INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                                    PredictionId INTEGER ,
                                    CfType varchar,
                                    Label varchar,
                                    Score REAL,
                                    FOREIGN KEY(PredictionId) REFERENCES Predictions(PredictionId));'''
        create_cfs = '''
            CREATE TABLE CFs (
                CfId INTEGER ,
                FeatureName varchar ,
                NewValue varchar ,

                FOREIGN KEY(CfId) REFERENCES Prediction_CFs(CfId)
                );'''
        tables_schema = [create_classifiers, create_predictions, create_prediction_cfs, create_cfs]
        for schema in tables_schema:
            self.cur.execute(schema)

    def init_instances_table(self):
        self.instances.instances_relation.to_sql('Instances', self.db, if_exists='append', index=True)

    def get_prediction_proba(self, classifier_id, instances):
        (model, enc, model_type) = self.models[classifier_id]
        if model_type == 'Neural_Cec':
            p_one = model(torch.FloatTensor(enc.transform(instances).values))
            p_zero = 1 - p_one
            return torch.cat((p_zero, p_one), dim=1).detach().numpy()
        elif model_type == 'Neural_Dice':
            p_one = model(torch.FloatTensor(enc.get_ohe_min_max_normalized_data(instances).values))
            p_zero = 1 - p_one
            return torch.cat((p_zero, p_one), dim=1).detach().numpy()
        else:
            return model.predict_proba(instances)


    def init_predictions_table(self):
        for classifier_id in range(len(self.models)):
            prediction_proba = self.get_prediction_proba(classifier_id, self.instances.data)
            labels, scores = self.get_label_scores(prediction_proba)
            predictions_tuples = {'ClassifierId': [classifier_id] * len(self.instances),
                                  'InstanceId': self.instances.data.index,
                                  'Label': labels,
                                  'Score': scores}
            predictions = pd.DataFrame(data=predictions_tuples)
            predictions.to_sql('Predictions', self.db, if_exists='append', index=False)

    def get_prediction_id_label(self, instance_id, classifier_id):
        return self.cur.execute(f'''select PredictionId ,Label 
                                    from Predictions 
                                    where InstanceId={instance_id}
                                    and ClassifierId={classifier_id}''').fetchone()

    def get_label(self, instance_id, is_sql=False):
        """
        Args:
          instance_id: instance ID.
        Returns:
          Ground truth of that instance
        """
        if is_sql:
            return self.cur.execute(f'''select {self.instances.outcome_name} 
                                    from Instances 
                                    where InstanceId={instance_id}''').fetchone()[0]
        else:
            return self.instances.get_label(instance_id)

    def add_view(self, cf_type: str, prediction_query: str, cfs_query):
        if isinstance(cfs_query, str):
            func_name, args = querylang_parser.parse_demo_query(cfs_query)
        else:
            func_name = cfs_query[0].__name__ if cfs_query is not None else 'No_Con'
        if not is_valid_view(cf_type, func_name):
            print(f'{cf_type} not support {func_name} query')
            return
        self.suffix += 1
        lazy_cf = CFsView(self, cf_type, prediction_query, cfs_query, self.processes)
        lazy_cf.print_relations_names()
        self.cf_views[lazy_cf.suffix] = lazy_cf

    def execute(self, query, parallel=True, verbose=False):
        tables = querylang_parser.extract_tables(query)
        to_execute = True
        for table in tables:
            if table in ['Instances', 'Prediction_CFs', 'Predictions', 'CFs', 'Classifiers', 'Reconstruct_CFs']:
                continue
            suffix = int(table.split('_')[-1])
            view = self.cf_views[suffix]
            if not view.is_computed:
                view.get_relations(parallel=parallel, verbose=verbose)
            if view.cfs is None or view.prediction_cfs is None:
                print("Sorry, no cf found, can't execute:")
                print(sqlparse.format(query, reindent=True, keyword_case='upper'))
                to_execute = False

        if to_execute:
            return pd.read_sql_query(query, self.db)

    def get_expressions(self, query):
        tables = querylang_parser.extract_tables(query)
        expressions = []
        for table in tables:
            if table in ['Instances', 'Prediction_CFs', 'Predictions', 'CFs', 'Classifiers', 'Reconstruct_CFs']:
                continue

            suffix = int(table.split('_')[-1])
            view = self.cf_views[suffix]
            if not view.is_computed:
                expressions.extend(view.get_expressions())

        return expressions

    def create_cfs_view(self, cf_type: str, prediction_query: str, cfs_query=None):
        if cf_type not in ['DiverseCFs', 'CecCFs', 'GrowingSpheresCFs']:
            raise NotImplementedError(f'{cf_type} not implement yet')
        else:
            self.add_view(cf_type, prediction_query, cfs_query)

    def get_label_scores(self, prediction_proba):
        index = np.argmax(prediction_proba, axis=1)
        scores = np.amax(prediction_proba, axis=1)
        labels = [self.instances.classes[i] for i in index]
        return labels, scores

    def get_instances(self, query: str):
        return pd.read_sql_query(query, self.db)['InstanceId']

    def get_predictions(self, prediction_query):
        return pd.read_sql_query(prediction_query, self.db)['PredictionId']

    def get_next_cf_id(self):
        query_last_cf_id = f'''select MAX(CfId) from Prediction_CFs'''
        idx = self.cur.execute(query_last_cf_id).fetchone()[0]
        return idx + 1 if idx else 1

    def get_changed_values(self, cf_alg: cf_generator.CfGenerator, instance_id: int, cfs_query):
        prediction_id, prediction_label = self.get_prediction_id_label(instance_id, cf_alg.classifier_id)
        instance = self.instances.get_instance(instance_id)
        conditions = cfs_query
        # cfs dataframe
        return cf_alg.get_cfs(instance, to_class=1 - prediction_label, conditions=conditions)

    def insert_to_db(self, cf_alg: cf_generator.CfGenerator, instance, cfs: Union[pd.DataFrame, None], prediction_id):
        """Insert the CFs to the database.

        Args:
          cf_alg: counterfactuals generator object.
          instance: pandas dataframe.
          cfs: counterfactuals as pandas dataframe.
          prediction_id: prediction id.
        Returns:
          two pandas dataframes that added to 'CFs' and 'Prediction_CFs' relations respectively.
        """
        if cfs is None:
            return None, None

        n_cfs = cfs['CfId'].nunique()
        new_cf_id = self.get_next_cf_id()
        all_values = pd.concat([instance] * n_cfs, ignore_index=True)
        for _, cf_id, f_name, new_val in cfs.itertuples():
            all_values.loc[cf_id, f_name] = new_val

        prediction_cfs_proba = self.get_prediction_proba(cf_alg.classifier_id, all_values)
        labels, scores = self.get_label_scores(prediction_cfs_proba)

        # prediction_cfs relation
        prediction_cfs_tuples = {'PredictionId': [prediction_id] * len(labels),
                                 'CfType': [cf_alg.cf_type] * len(labels),
                                 'Label': labels,
                                 'Score': scores}
        prediction_cfs = pd.DataFrame(data=prediction_cfs_tuples)
        prediction_cfs.to_sql('Prediction_CFs', self.db, if_exists='append', index=False)

        # cfs relation
        cfs['CfId'] = cfs['CfId'] + new_cf_id
        cfs.to_sql('CFs', self.db, if_exists='append', index=False)

        # reconstruct_cfs relation
        all_values.insert(0, 'CfId', range(new_cf_id, n_cfs+new_cf_id))
        all_values.to_sql('Reconstruct_CFs', self.db, if_exists='append', index=False)

        return cfs, prediction_cfs

    def get_cf(self, cf_alg: cf_generator.CfGenerator, instance_id: int, cfs_query):
        cfs = self.get_changed_values(cf_alg, instance_id, cfs_query)
        instance = self.instances.get_instance(instance_id)
        prediction_id, prediction_label = self.get_prediction_id_label(instance_id, cf_alg.classifier_id)
        return self.insert_to_db(cf_alg, instance, cfs, prediction_id)

    # return dictionary of CF generation name --> basic CF algorithm for each model
    def init_cf_generation(self):
        # dice
        self.init_dice()
        self.init_cec()
        self.init_growingspheres()

    # TODO change the assumption the constant at dice
    def init_dice(self):

        dice_params = {
                       'method': "random",
                       'dataframe': self.instances.instances_relation,
                       'continuous_features': self.instances.continuous_features,
                       'total_CFs': 3
                       }
        for classifier_id, (model, enc, model_type) in enumerate(self.models):
            if model_type == 'Neural_Cec':
                continue
            elif model_type == 'Neural_Dice':
                dice_params['backend'] = "PYT"
            else:
                dice_params['backend'] = "sklearn"
            dice = cf_generator.DiceCfGenerator(model, classifier_id, enc, model_type, self.instances.outcome_name, **dice_params)
            self.CF_generations['DiverseCFs'][classifier_id]=dice

    def init_growingspheres(self):
        gs_params = {}
        for classifier_id, (model, enc, model_type) in enumerate(self.models):
            if model_type == 'Neural_Dice':
                continue
            gs = cf_generator.GrowingSpheresCfGenerator(model, classifier_id, enc, model_type, self.instances.outcome_name, **gs_params)
            self.CF_generations['GrowingSpheresCFs'][classifier_id] = gs

    def init_cec(self):

        cec_params = {'features': self.instances.feature_names}
        for classifier_id, (model, enc, model_type) in enumerate(self.models):
            if model_type == 'Trees':
                cec = cf_generator.CecTreeCfGenerator(model, classifier_id, enc, model_type, self.instances.outcome_name, **cec_params)
                self.CF_generations['CecCFs'][classifier_id] = cec
            elif model_type == 'Linear':
                cec = cf_generator.CecLinearCfGenerator(model, classifier_id, enc, model_type, self.instances.outcome_name, **cec_params)
                self.CF_generations['CecCFs'][classifier_id] = cec
            elif model_type == 'Neural_Cec':
                cec = cf_generator.CecNeuralCfGenerator(model, classifier_id, enc, model_type, self.instances.outcome_name, **cec_params)
                self.CF_generations['CecCFs'][classifier_id] = cec
            else:
                continue

    def reconstruct_cf(self, cf_id: int, highlight=True):
        """ Reconstruct counterfactual.
        Args:
          cf_id: counterfactual ID.
          highlight: boolean variable that determine if to highlight the changed features in the counterfactual
        Returns:
          the reconstruct counterfactual that identified by cf_id.
        """
        reconstruct_q = f'''
        SELECT *
        FROM  Reconstruct_CFs
        WHERE Reconstruct_CFs.CfId = {cf_id}
        '''
        features_q = f'''
        SELECT CFs.FeatureName
        FROM  CFs
        Where CFs.CfId = {cf_id}
        '''
        reconstruct = self.execute(reconstruct_q, parallel=False, verbose=False)
        features = self.execute(features_q, parallel=False, verbose=False)
        features = list(features.T.values[0])

        return reconstruct.style.set_properties(**{'background-color': 'orange', 'subset': features}) if highlight else reconstruct

    def reconstruct(self, cf_ids, highlight=True):
        """ Reconstruct list of counterfactual.
        Args:
          cf_ids: list of counterfactual ID.
          highlight: boolean variable that determine if to highlight the changed features in the counterfactuals
        Returns:
          the reconstruct counterfactuals that identified by cf_ids.
        """
        instances_q = f'''
        SELECT DISTINCT CFs.CfId, Instances.*
        FROM  CFs,Prediction_CFs, Predictions,Instances
        Where CFs.CfId = Prediction_CFs.CfId
        and Predictions.PredictionId = Prediction_CFs.PredictionId 
        and Instances.InstanceId = Predictions.InstanceId
        and CFs.CfId In {tuple(cf_ids)}
        '''
        original_instances = self.execute(instances_q, parallel=True, verbose=False)
        original_instances = original_instances.drop(['InstanceId', self.instances.outcome_name], axis=1)
        dfs = [self.reconstruct_cf(idx, highlight=False) for idx in cf_ids]
        cfs = pd.concat(dfs, ignore_index=True)
        if highlight:
            return cfs.style.apply(highlight_diff, axis=None, original_instances=original_instances)
        return cfs

    def get_arg_for_perallelism(self, cf_alg: cf_generator.CfGenerator, instance_id: int, query: str):
        prediction_id, prediction_label = self.get_prediction_id_label(instance_id, cf_alg.classifier_id)
        instance = self.instances.get_instance(instance_id)
        return cf_alg, 1 - prediction_label, instance, query

    def get_instance_classifier_id(self, prediction_id):
        return self.cur.execute(f'''select ClassifierId ,InstanceId, Label 
                                    from Predictions 
                                    where PredictionId={prediction_id}''').fetchone()




class CFsView:
    # self, cf_type: str, instances_query: str, cfs_query: str
    def __init__(self, querylang: QueryLang, cf_type: str, prediction_query: str, cfs_query, processes=2):
        self.processes = processes
        self.querylang = querylang
        self.suffix = querylang.suffix
        self.cf_type = cf_type
        self.prediction_query = prediction_query
        self.prediction_query_type = self.get_prediction_query(prediction_query)
        # self.cfs_query = cfs_query if cfs_query is not None else '''SELECT CFs.* FROM CFs '''
        # expression
        if isinstance(cfs_query, str):
            self.cfs_func, self.cfs_args = querylang_parser.parse_demo_query(cfs_query)
        else:
            self.cfs_func, self.cfs_args = (cfs_query[0], cfs_query[1]) if cfs_query is not None else (queries.No_Con, {})
        self.is_computed = False
        self.cfs = None
        self.prediction_cfs = None
        self.times = pd.DataFrame(columns=['ViewId', 'ClassifierId', 'InstanceId', 'Success', 'Time'])

    def print_relations_names(self):
        print("Your relations names are: {0} and {1}".format(self.get_cf_name(), self.get_prediction_cf_name()))

    def get_cf_name(self):
        return f'my_cfs_{self.suffix}'

    def get_prediction_cf_name(self):
        return f'my_prediction_cfs_{self.suffix}'

    def get_prediction_cf_from_db(self):
        q = f""" SELECT prediction_CFs.*
                from prediction_CFs 
                where prediction_CFs.CfId IN (SELECT CfId FROM {self.get_cf_name()}) """
        return pd.read_sql_query(q, self.querylang.db)

    def get_expressions(self):
        predictions = self.querylang.get_predictions(self.prediction_query)
        expressions = []
        for prediction_id, cfs_query in product(predictions, [(self.cfs_func, self.cfs_args)]):
            classifier_id, instance_id, prediction_label = self.querylang.get_instance_classifier_id(prediction_id)
            cf_alg = self.querylang.CF_generations[self.cf_type][classifier_id]
            instance = self.querylang.instances.get_instance(instance_id)
            z3_expression = self.cfs_func(cf_alg.encoder, instance, **self.cfs_args, is_dnf=False, is_z3_format=True)
            expressions.append(PredictionsExample(prediction_id=prediction_id,
                                                  instance_id=instance_id,
                                                  z3_expression=z3_expression,
                                                  constraints=cfs_query,
                                                  label=prediction_label,
                                                  view_id=self.suffix))
        return expressions


    def get_relations(self, parallel=True, verbose=False):
        if verbose:
            print(f'Evaluate {self.get_cf_name()} and {self.get_prediction_cf_name()} in type {self.cf_type}:')
            print()
            print('with the following prediction query')
            if self.prediction_query_type is None:
                print(sqlparse.format(self.prediction_query, reindent=True, keyword_case='upper'))
            else:
                print(self.prediction_query_type)
            print()
            #print('and with the following cfs query')
            #print(sqlparse.format(self.cfs_query, reindent=True, keyword_case='upper'))
            print('and with the following query type')
            print(self.cfs_func.__name__)
            print('-' * 60)
        predictions = self.querylang.get_predictions(self.prediction_query)
        dfs = []
        total = len(predictions)
        if parallel:
            args = []
            for prediction_id, cfs_query in product(predictions, [(self.cfs_func, self.cfs_args)]):
                classifier_id, instance_id, prediction_label = self.querylang.get_instance_classifier_id(prediction_id)
                cf_alg = self.querylang.CF_generations[self.cf_type][classifier_id]
                instance = self.querylang.instances.get_instance(instance_id)
                args.append((cf_alg, 1 - prediction_label, prediction_id, instance_id, instance, cfs_query))

            with ProcessPoolExecutor(max_workers=self.processes) as pool:
                with tqdm(total=total, leave=False) as progress:
                    futures = []

                    for arg in args:
                        future = pool.submit(get_changes_multiprocess, arg)
                        future.add_done_callback(lambda p: progress.update())
                        futures.append(future)

                    output = []
                    for future in futures:
                        result = future.result()
                        output.append(result)

            for cf_alg, instance, cfs, prediction_id, total_time in output:
                dfs.append(self.querylang.insert_to_db(cf_alg, instance, cfs, prediction_id))
                index = instance.index[0]
                success = 0 if cfs is None else 1
                self.times.loc[len(self.times)] = [self.suffix, cf_alg.classifier_id, index, success, total_time]

        else:
            for prediction_id in tqdm(predictions, leave=False, total=total):
                classifier_id, instance_id, prediction_label = self.querylang.get_instance_classifier_id(prediction_id)
                cf_alg = self.querylang.CF_generations[self.cf_type][classifier_id]
                start_time = time.time()
                cfs = self.querylang.get_changed_values(cf_alg, instance_id, (self.cfs_func, self.cfs_args))
                total_time = time.time() - start_time
                instance = self.querylang.instances.get_instance(instance_id)

                output = self.querylang.insert_to_db(cf_alg, instance, cfs, prediction_id)
                dfs.append(output)
                index = instance.index[0]
                success = 0 if cfs is None else 1
                self.times.loc[len(self.times)] = [self.suffix, cf_alg.classifier_id, index, success, total_time]

        cfs, prediction_cfs = list(map(list, zip(*dfs)))
        found = sum(x is not None for x in cfs)
        if verbose:
            print('-' * 60)
            print(f'Found {found} CFs out of {total} requested')
            print('-' * 60)
        if found > 0:
            cfs = pd.concat(cfs, ignore_index=True)
            cfs.to_sql(self.get_cf_name(), self.querylang.db, if_exists='append', index=False)
            prediction_cfs = self.get_prediction_cf_from_db()
            prediction_cfs.to_sql(self.get_prediction_cf_name(), self.querylang.db, if_exists='append', index=False)
        else:
            cfs, prediction_cfs = None, None
        self.cfs = cfs
        self.prediction_cfs = prediction_cfs
        self.is_computed = True
        return cfs, prediction_cfs

    def get_prediction_query(self, prediction_query):
        try:
            size = int(prediction_query.split('<=')[-1].strip())
            for ClassifierId in range(len(self.querylang.models)):
                if get_prediction_query(size, label=0, prediction=0, target=self.querylang.instances.outcome_name, classifier_id=ClassifierId) == prediction_query:
                    return f'True Negative Classifier {ClassifierId}'
                elif get_prediction_query(size, label=1, prediction=1, target=self.querylang.instances.outcome_name, classifier_id=ClassifierId) == prediction_query:
                    return f'True Positive Classifier {ClassifierId}'
                elif get_prediction_query(size, label=0, prediction=1, target=self.querylang.instances.outcome_name, classifier_id=ClassifierId) == prediction_query:
                    return f'False Positive Classifier {ClassifierId}'
                elif get_prediction_query(size, label=1, prediction=0, target=self.querylang.instances.outcome_name, classifier_id=ClassifierId) == prediction_query:
                    return f'False Negative Classifier {ClassifierId}'
        except:
            return None


def get_changes_multiprocess(args):
    cf_alg, to_class, prediction_id, instance_id, instance, cfs_query = args
    conditions = cfs_query
    start_time = time.time()
    cfs = cf_alg.get_cfs(instance, to_class, conditions)
    total_time = time.time()- start_time
    return cf_alg, instance, cfs, prediction_id, total_time


class UserConfigValidationException(Exception):
    """An exception indicating that some user configuration is not valid.
    :param exception_message: A message describing the error.
    :type exception_message: str
    """
    _error_code = "Invalid Configuration"
