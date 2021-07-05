# Required Libraries
import argparse
import os

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import skew, kurtosis, probplot, randint, uniform, t, sem, loguniform
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings(action='ignore')

COL_GROUPS = {
    "Demographic factor": ["AGE", "SEX", "BMI", "BMD"],
    "Radiologic factor": ["INITIAL K-L GRADE"],
    "Underlying disease": ["HYPERTENSION", "DIABETES", "OTHER_DISORDERS"],
    "Occupation": ["OCCUPATION"],
    "Progression rate": ["PROGRESSION RATE"]
}

MODELS = {
    "lr": LogisticRegression,
    "rf": RandomForestClassifier,
    "gbm": GradientBoostingClassifier
}
# LR_PARAMS = {"penalty": ["l1", "l2"], "C":loguniform(1e-2, 1e2)}
RF_PARAMS = {"max_depth": randint(1, 15), "n_estimators": randint(100, 1000)}
GBM_PARAMS = {
    "max_depth": randint(1, 15),
    "learning_rate": uniform(0.001, 0.3),
    "n_estimators": randint(100, 1000)
}
PARAMS = {"lr": None, "rf": RF_PARAMS, "gbm": GBM_PARAMS}


class DataGroups:
    def __init__(self, col_groups):
        # Demographic factor
        self.x_demo = COL_GROUPS["Demographic factor"]
        # Radiologic factor
        self.x_radio = COL_GROUPS["Radiologic factor"]
        # Demographic factor + Radiologic factor
        self.x_demo_radio = COL_GROUPS["Demographic factor"] + COL_GROUPS[
            "Radiologic factor"]
        # Demographic factor + Radiologic factor + Underlying disease
        self.x_demo_radio_under = COL_GROUPS["Demographic factor"] + COL_GROUPS[
            "Radiologic factor"] + COL_GROUPS["Underlying disease"]
        # Demographic factor + Radiologic factor + Occupation
        self.x_demo_radio_occu = COL_GROUPS["Demographic factor"] + COL_GROUPS[
            "Radiologic factor"] + COL_GROUPS["Occupation"]
        # Demographic factor + Radiologic factor + Underlying disease + Occupation
        self.x_total = COL_GROUPS["Demographic factor"] + COL_GROUPS[
            "Radiologic factor"] + COL_GROUPS[
                "Underlying disease"] + COL_GROUPS["Occupation"]

        self.x_col_list = [
            self.x_demo, self.x_radio, self.x_demo_radio,
            self.x_demo_radio_under, self.x_demo_radio_occu, self.x_total
        ]
        self.y_col = COL_GROUPS["Progression rate"]


def load_data(path):
    data = pd.read_csv(path, sep='\t', index_col=False)
    data['BMD'] = data['BMD'].replace('.', -1)
    return data


data_groups = DataGroups(COL_GROUPS)


def preprocess(data, x_col_list, y_col, test_size=0.3):
    train, test = train_test_split(data,
                                   stratify=data[y_col],
                                   test_size=test_size)
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []
    for x_col in x_col_list:
        x_train_list.append(train[x_col])
        x_test_list.append(test[x_col])
        y_train_list.append(train[y_col])
        y_test_list.append(test[y_col])
    return x_train_list, y_train_list, x_test_list, y_test_list


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t._ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def train(data, model_cls, params=None, random_search=True, epochs=10):
    result_dic = defaultdict(list)
    model_dict = {}
    model_dict_final = {}
    for epoch in range(epochs):
        x_train_list, y_train_list, x_test_list, y_test_list = preprocess(
            data, data_groups.x_col_list, data_groups.y_col)
        for i in range(len(x_train_list)):
            if params is not None:
                if random_search:
                    model = RandomizedSearchCV(model_cls(),
                                               params,
                                               n_iter=5,
                                               n_jobs=-1)
                else:
                    model = model_cls().set_params(**params)
            else:
                model = model_cls()
            model.fit(x_train_list[i].values, y_train_list[i].values.ravel())
            model_dict[(epoch, i)] = model
            result_dic[f'train_{i}'].append(
                model.score(x_train_list[i], y_train_list[i]))
            result_dic[f'test_{i}'].append(
                model.score(x_test_list[i], y_test_list[i]))
    return model_dict, result_dic


def get_ci(result_dic):
    mean_list = []
    for key in result_dic.keys():
        print(key, mean_confidence_interval(result_dic[key]))
        if 'test' in key:
            mean_list.append(mean_confidence_interval(result_dic[key])[0])
    return mean_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model type")
    parser.add_argument("--path", default="./data.csv", help="data path")
    args = parser.parse_args()
    path = args.path
    model = MODELS[args.model]
    params = PARAMS[args.model]
    data = load_data(path)
    model_dict, result_dic = train(data, model, params)
    get_ci(result_dic)

