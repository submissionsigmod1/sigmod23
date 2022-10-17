import sqlparse
from sqlparse.utils import imt
from sqlparse.sql import Where, Identifier, Comparison, Parenthesis, IdentifierList
import sqlparse.tokens as T
from sql_metadata import Parser

import queries


def parse_parenthesis(token: Parenthesis):
    idx = 0
    condition = ""
    exp = []
    n_atoms = 0
    closest_keyword = lambda tk: tk.match(sqlparse.tokens.Keyword, ('AND', 'OR'))
    while idx is not None:
        idx, comparison = token.token_next_by(idx=idx, i=Comparison)
        if idx:
            exp.append(parse_comparaison(comparison))
            condition += f' y_{n_atoms} '
            n_atoms += 1
            idx, grammar = token._token_matching(closest_keyword, start=idx + 1)
            if grammar:
                condition += '&' if grammar.value.upper() == 'AND' else '|'
    return queries.Q_demo, {'exp': condition, 'to_d': exp}


def parse(token):
    if isinstance(token, Parenthesis):
        return parse_parenthesis(token)
    elif isinstance(token, Comparison):
        return parse_comparaison(token)


def transform_dice_conditions(conditions):
    permitted_range = {}
    features_to_vary = set()
    for x in conditions:
        # CFs.FeatureName = value
        if isinstance(x, tuple):
            col = x[2]
            features_to_vary.add(col)
        elif isinstance(x, list):
            col = x[0][2]
            min_val = x[1][2]
            max_val = x[2][2]
            permitted_range[col] = [min_val, max_val]
            features_to_vary.add(col)
    return list(features_to_vary), permitted_range


def get_value(identifier):
    if imt(identifier, t=T.Literal):
        val = identifier.value
        # remove commas from start and end of the string
        if isinstance(val, str):
            val = eval(val)
        return val,
    return identifier.tokens[0], identifier.get_name()


def align_condition(left, sign, right):
    flip_signs = {'=': '==', '!=': "!=", '<=': '>', '>=': '<', '<': '>=', '>': '<=', '==': '=='}
    if sign == '=':
        sign = '=='
    if len(left) == 1:
        left, right = right, left
        sign = flip_signs[sign]
    exp = {'f1': left[1], 'sign': sign}
    if len(right) == 1:
        exp['f2'] = (right[0],)
    else:
        exp['f2'] = right[0].value, right[1]
    return exp


def parse_comparaison(token: Comparison):
    # first identifier
    idx, identifier = token.token_next_by(t=T.Literal, i=Identifier)
    left = get_value(identifier)

    # sign of comparison
    _, comparison = token.token_next_by(t=T.Comparison)
    sign = comparison.value

    # second identifier
    idx, second_identifier = token.token_next_by(idx=idx, t=T.Literal, i=Identifier)
    right = get_value(second_identifier)
    return align_condition(left, sign, right)


def get_dice_conditions(sql: str):
    parsed = sqlparse.parse(sql)[0]
    _, where = parsed.token_next_by(i=Where)
    if where is None:
        return transform_dice_conditions([])
    idx = 0
    conditions = []
    # search closest Comparison/Parenthesis
    closest_condition = lambda tk: imt(tk, i=(Parenthesis, Comparison))
    while idx is not None:
        idx, token = where._token_matching(closest_condition, start=idx + 1)
        if idx:
            conditions.append(parse(token))
    # return conditions
    return transform_dice_conditions(conditions)





def transform_cec(conditions):
    o = []
    for conjunction in conditions:
        permitted_range = {}
        features_to_vary = set()
        features_not_to_vary = set()
        for x in conjunction:
            if isinstance(x, tuple):
                col, sign, val = x
                # sign can be '=' or '!='
                if sign == '=':
                    features_to_vary.add(val)
                else:  # sign =='!='
                    features_not_to_vary.add(val)

            elif isinstance(x, list):
                # assume x[0] = ('FeatureName', '=', 'col name')
                col = x[0][2]
                features_to_vary.add(col)
                if len(x) == 2:
                    _, sign, val = x[1]
                    if sign == "=":
                        permitted_range[col] = val
                    # ignore the difference between "<"(">") and "<="(">=")
                    elif sign in [">", ">="]:
                        permitted_range[col] = [val, float('inf')]
                    elif sign in ["<", "<="]:
                        permitted_range[col] = [float('-inf'), val]

                elif len(x) == 3:
                    # assume x[0] = ('FeatureName', '=', 'col name')
                    # assume x[0] = ('FeatureName', '>='('>'), 'col name')
                    # assume x[0] = ('FeatureName', '<='('>'), 'col name')
                    min_val = x[1][2]
                    max_val = x[2][2]
                    permitted_range[col] = [min_val, max_val]
        o.append((features_to_vary, features_not_to_vary, permitted_range))
    return o


def is_subselect(parsed):
    if not parsed.is_group:
        return False
    for item in parsed.tokens:
        if item.ttype is T.DML and item.value.upper() == 'SELECT':
            return True
    return False


def extract_from_part(parsed):
    from_seen = False
    for item in parsed.tokens:
        if from_seen:
            if is_subselect(item):
                yield from extract_from_part(item)
            elif item.value.upper() in ["ORDER BY", "GROUP BY"]:
                return
            elif item.ttype is T.Keyword:
                #print('key',item)
                #return
                continue
            else:
                yield item
        elif item.ttype is T.Keyword and item.value.upper() == 'FROM':
            from_seen = True


def extract_table_identifiers(token_stream):
    for item in token_stream:
        if isinstance(item, IdentifierList):
            for identifier in item.get_identifiers():
                yield identifier.tokens[0].value
        elif isinstance(item, Identifier):
            yield item.tokens[0].value



def extract_tables(sql):
    return Parser(sql).tables


Implies_demo = '''
SELECT *
FROM Reconstruct_CFs AS RC,
     Prediction_CFs as PC,
     Predictions AS P,
     Instances AS I
WHERE RC.CfId = PC.CfId
  AND PC.PredictionId = P.PredictionId
  AND I.InstanceId = P.InstanceId
  AND (RC.gender = 5 OR RC.education != I.education)
'''



def parse_demo_query(query: str):
    conditions = []
    p = Parser(query)
    parsed = sqlparse.parse(query)[0]
    if 'IN' in query:
        for token in p._not_parsed_tokens:
            if token.is_in_parenthesis and token.value != ',':
                conditions.append(eval(token.value))
        return queries.OR, {'features': conditions}
    elif 'Reconstruct_CFs' in query:
        where = parsed.token_next_by(i=Where)[1]
        par = where.token_next_by(i=sqlparse.sql.Parenthesis)[1]
        return parse_parenthesis(par)
    else:
        _, where = parsed.token_next_by(i=Where)
        idx = 0
        conditions = []
        # search closest Comparison/Parenthesis
        closest_condition = lambda tk: imt(tk, i=(Parenthesis, Comparison))
        while idx is not None:
            idx, token = where._token_matching(closest_condition, start=idx + 1)
            if idx and token[0][-1].value != 'CfId':
                conditions.append(eval(token[-1].value))
        return queries.AND, {'features': conditions}


