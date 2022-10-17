import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from QUERYLANG import Instances
from preprocessing import MultiColumnEncoder

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":
    url = 'data/Adult Income/adult_data.csv'
    df = pd.read_csv(url)
    numerical = ["age", "hours_per_week"]
    categorical = ['education', 'gender', 'marital_status', 'occupation', 'race', 'workclass']
    GroundTruth = "income"
    feature_names = [*categorical, *numerical,GroundTruth]
    df = df[feature_names]

    target = df[GroundTruth]
    train_dataset, test_dataset, y_train, y_test = train_test_split(df,
                                                                    target,
                                                                    test_size=0.3,
                                                                    random_state=0,
                                                                    stratify=target)
    x_train = train_dataset.drop(GroundTruth, axis=1)
    x_test = test_dataset.drop(GroundTruth, axis=1)

    rf = Pipeline(steps=[('le', MultiColumnEncoder(target=True)),
                         ('classifier', RandomForestClassifier(n_estimators=10,
                                                               max_depth=5,
                                                               min_samples_split=5,
                                                               random_state=1,
                                                               n_jobs=-1))])

    linear = Pipeline(steps=[('le', MultiColumnEncoder(target=True)),
                             ('classifier', SGDClassifier(loss='log_loss',
                                                          penalty="l2",
                                                          max_iter=1000,
                                                          tol=1e-3))])
    rf.fit(x_train, y_train)
    linear.fit(x_train, y_train)

    print(f'Linear: train acc = {linear.score(x_train, y_train)} test acc = {linear.score(x_test, y_test)}')
    print(f'Random Forest({10} trees): train acc = {rf.score(x_train,y_train)} test acc = {rf.score(x_test,y_test)}')

    models = [(rf, rf[0], 'Trees'), (linear, linear[0], 'Linear')]
    instances = Instances(data=x_train, labels=y_train, favorite_class=1)

    models_file = open('models/Adult Income/adult_models.pkl', 'wb')
    pickle.dump(models, models_file)

    instances_file = open('Instances/Adult Income/adult_train_instances.pkl', 'wb')
    pickle.dump(instances, instances_file)

    test_instances = Instances(data=x_test, labels=y_test, favorite_class=1)
    test_instances_file = open('Instances/Adult Income/adult_test_instances.pkl', 'wb')
    pickle.dump(test_instances, test_instances_file)
