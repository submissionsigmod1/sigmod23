from itertools import product

import re

from dnf_converter.dnf_converter import Logic_expression

No_constraints = '''SELECT CFs.* FROM CFs '''

Q_dummy = ''' 
SELECT *
FROM CFs AS C,
WHERE C.FeatureName = 'x2'
  AND C.NewValue >= 1.2
'''

prediction_q = ''' SELECT T.PredictionId
FROM (
    SELECT Predictions.PredictionId , ROW_NUMBER() OVER(PARTITION BY Predictions.ClassifierId) AS rank
    FROM Instances, Predictions
    WHERE Instances.InstanceId = Predictions.InstanceId
      AND Instances.{Target} = {label} and  Predictions.Label = {prediction}
      AND Predictions.ClassifierId = {ClassifierId}) as T
WHERE T.rank <= {size}
'''


def get_z3_str_with_val(encoder, feature_name, sign, val):
    return f'y_{feature_name} {sign} {encoder.label_encoder[feature_name][val]}' if feature_name in encoder.columns \
        else f'y_{feature_name} {sign} {val}'


def get_z3_str(encoder, instance, feature_name, sign):
    return f'y_{feature_name} {sign} {encoder.label_encoder[feature_name][instance[feature_name].values[0]]}' if feature_name in encoder.columns \
        else f'y_{feature_name} {sign} {instance[feature_name].values[0]}'


def get_prediction_queries(size, target, classifier_ids):
    return [get_prediction_query(size, label, prediction, target, classifier_id) for label, prediction, classifier_id in
            product([0, 1], [0, 1], classifier_ids)]


def get_prediction_query(size, label, prediction, target, classifier_id):
    args = {'size': size, 'label': label, 'prediction': prediction, 'Target': target, 'ClassifierId': classifier_id}
    return prediction_q.format(**args)


def get_pattern(d):
    return re.compile(r'\b(' + '|'.join(d.keys()) + r')\b')


def DUMMY(val):
    exp = 'C_x2 & y_x2'
    d = {
        'y_x2': f'(y_x2 >= {val})'
    }
    pattern = get_pattern(d)
    dnfs = Logic_expression(exp).dnf_string.split('|')
    return [pattern.sub(lambda x: d[x.group()], a[1:-1]).split('&') for a in dnfs]


def get_expression(func, **args):
    return func(**args)


def get_dnfs(exp, d=None):
    if exp == []:
        return [[]]
    if d is not None:
        pattern = re.compile(r'\b(' + '|'.join(d.keys()) + r')\b')
        dnfs = Logic_expression(exp).dnf_string.split('|')
        return [pattern.sub(lambda x: d[x.group()], a[1:-1]).split('&') for a in dnfs]
    else:
        dnfs = Logic_expression(exp).dnf_string.split('|')
        return [a[1:-1].split('&') for a in dnfs]


def No_Con(encoder, instance, is_dnf=False, is_z3_format=False):
    return get_exp([], None, is_dnf)


def OR(encoder, instance, features, is_dnf=False, is_z3_format=False):
    if not isinstance(features, list):
        features = [features]
    exp = ' | '.join([f'C_{f}' for f in features])
    if is_z3_format:

        # exp = ', '.join([f'y_{f} != {instance[encoder.features.get_loc(f)]}' for f in features])

        exp = ', '.join([get_z3_str(encoder, instance, feature_name, '!=') for feature_name in features])
        exp = f'Or({exp})'
    return get_exp(exp, None, is_dnf)


def AND(encoder, instance, features, is_dnf, is_z3_format=False):
    if not isinstance(features, list):
        features = [features]
    exp = '&'.join([f' C_{f} ' for f in features])
    if is_z3_format:
        exp = ', '.join([ get_z3_str(encoder, instance, feature_name, '!=') for feature_name in features])
        exp = f'And({exp})'
    return get_exp(exp, None, is_dnf)
    #return get_dnfs(exp)


def NOT_AND(encoder, instance, features, is_dnf, is_z3_format=False):
    exp = '|'.join([f' ~C_{f} ' for f in features])

    if is_z3_format:

        # exp = ', '.join([f'y_{f} != {instance[encoder.features.get_loc(f)]}' for f in features])

        exp = ', '.join([get_z3_str(encoder, instance, feature_name, '==') for feature_name in features])
        exp = f'Or({exp})'
    return get_exp(exp, None, is_dnf)


def AND_NOT(encoder, instance, features, is_dnf,is_z3_format=False):
    exp = '&'.join([f' ~C_{f} ' for f in features])
    if is_z3_format:

        # exp = ', '.join([f'y_{f} != {instance[encoder.features.get_loc(f)]}' for f in features])

        exp = ', '.join([get_z3_str(encoder, instance, feature_name, '==') for feature_name in features])
        exp = f'And({exp})'
    return get_exp(exp, None, is_dnf)


def IMPLIES(encoder, instance, f1, f2, is_dnf, is_z3_format=False):
    exp = f' C_{f2} | ~C_{f1}'
    if is_z3_format:
        u = get_z3_str(encoder, instance, f1, '==')
        v = get_z3_str(encoder, instance, f2, '!=')
        exp = f'Or({u}, {v})'
    return get_exp(exp, None, is_dnf)


def Q7(encoder, instance, f1, f2, is_dnf, is_z3_format=False):
    if hasattr(encoder, 'features'):
        idx = encoder.features.get_loc(f1)
    else:
        idx = encoder.feature_names.index(f1)
    exp = f' C_{f1} & y_{f1}  & ~C_{f2}'
    try:
        d = {
            f'y_{f1}': f'(y_{f1} > {instance[idx]})'
        }
    except Exception:
        d = {
            f'y_{f1}': f'(y_{f1} > {instance[f1].values[0]})'
        }
    if is_z3_format:
        u = get_z3_str(encoder, instance, f1, '>')
        v = get_z3_str(encoder, instance, f2, '==')
        exp = f'And({u}, {v})'
        d = None
    return get_exp(exp, d, is_dnf)


def Q8(encoder, instance, f1, f2, vals, is_dnf, is_z3_format=False):
    if not isinstance(vals, list):
        vals = [vals]
    exp = ' | '.join([f'y_{i}' for i, v in enumerate(vals)])
    exp = f'C_{f1} & C_{f2} & ' + f'({exp})'
    d = {
        f'y_{i}': f'(y_{f2} == {v})' for i, v in enumerate(vals)
    }
    if is_z3_format:
        u = get_z3_str(encoder, instance, f1, '!=')
        # v = get_z3_str(encoder, instance, f2, '!=')
        w = ', '.join([f'y_{f2} == {v}' for v in vals])
        #exp = f'And({u},{v},Or({w}))'
        exp = f'And({u},Or({w}))'
        d = None
    return get_exp(exp, d, is_dnf)


def Q9(encoder, instance, f1, min_val, max_val, is_dnf, is_z3_format=False):
    exp = f'C_{f1} & y_1 & y_2'
    d = {
        f'y_1': f'(y_{f1} >= {min_val})',
        f'y_2': f'(y_{f1} <= {max_val})'
    }
    if is_z3_format:
        u = f'y_{f1} >= {min_val}'
        v = f'y_{f1} <= {max_val}'
        exp = f'And({u},{v})'
        d = None
    return get_exp(exp, d, is_dnf)


def Q10(encoder, instance, f1, val, is_dnf, is_z3_format=False):
    exp = f'y_1'
    d = {
        f'y_1': f"(y_{f1} == '{val}')",
    }
    if is_z3_format:
        exp = get_z3_str_with_val(encoder, f1, '==', val)
        d = None
    return get_exp(exp, d, is_dnf)


def get_exp(exp, d=None, is_dnf=False):
    if exp == []:
        return [[]] if is_dnf else 'True'
    if d is None:
        if not is_dnf:
            return exp.replace('&', 'and ').replace('|', 'or').replace('~', 'not ')
        else:
            dnfs = Logic_expression(exp).dnf_string.split('|')
            return [a[1:-1].split('&') for a in dnfs]
    else:
        pattern = re.compile(r'\b(' + '|'.join(d.keys()) + r')\b')
        if not is_dnf:
            return pattern.sub(lambda x: d[x.group()], exp).replace('&', 'and ').replace('|', 'or').replace('~', 'not ')
        else:
            dnfs = Logic_expression(exp).dnf_string.split('|')
            return [pattern.sub(lambda x: d[x.group()], a[1:-1]).split('&') for a in dnfs]



