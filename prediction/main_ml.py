import urllib3
import pandas as pd
import numpy as np
import zipfile
import copy
import pickle
import os
from esig import tosig
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from os import listdir
from os.path import isfile, join
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix


def get_inputs():
    url = "https://storage.googleapis.com/challenge-2012-1.0.0.physionet.org/set-a.zip"
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=False)

    with open('data/input.zip', 'wb') as out:
        while True:
            data = r.read()
            if not data:
                break
            out.write(data)

    r.release_conn()
    zip_ref = zipfile.ZipFile("data/input.zip", 'r')
    zip_ref.extractall("data/")
    zip_ref.close()

    data = {}
    list_files = [f for f in listdir(
        "data/set-a") if isfile(join("data/set-a", f))]

    for f in list_files:
        df = pd.read_csv(join("data/set-a", f))
        patient_id = int(df.values[0, 2])
        data[patient_id] = df

    return data


def get_outputs():
    url = "https://storage.googleapis.com/challenge-2012-1.0.0.physionet.org/Outcomes-a.txt"
    data_df = pd.read_csv(url)
    data = {}

    for patient in data_df.values:
        patient_id = int(patient[0])
        data[patient_id] = patient[-1]

    return data


def download():
    X_dict, Y_dict = get_inputs(), get_outputs()

    X = []
    Y = []
    for patient_id in X_dict:
        X.append(X_dict[patient_id])
        Y.append(Y_dict[patient_id])

    print("Data for %s patients downloaded." % len(X))
    return X, Y


def split(X, Y, proportion=0.75):
    idx = int(len(X)*proportion)
    print("Dataset split in a training set of %s and testing set of %s patients." % (
        idx, len(X)-idx))
    return X[:idx], Y[:idx], X[idx:], Y[idx:]


def features_point(x):
    static, path = x
    maximums = np.max(path, axis=0)
    minimums = np.min(path, axis=0)
    last_observation = path[-1]

    return np.concatenate([static, maximums, minimums, last_observation])


def extract(X):
    return list(map(features_point, X))


def lead_lag(mylist):
    leadlist = np.concatenate([[mylist[0]], mylist])
    laglist = np.concatenate([mylist, [mylist[-1]]])
    return np.concatenate([leadlist, laglist], axis=1)


def add_time(mylist, init_time=0., total_time=1.):
    ans = [[init_time + xn * total_time /
            (len(mylist)-1)] + list(x) for (xn, x) in enumerate(mylist)]
    return np.array(ans)


def home_and_pen_off(mylist):
    ans = [list(x) + [1.] for x in mylist]
    last = list(ans[-1])
    last[-1] = 0.
    ans.append(last)
    ans.append([0 for item in last])
    return np.array(ans)


def refocus(path, centre):
    return np.concatenate((centre[::-1], path), axis=0)


def train(features, Y):
    classifier = RandomForestClassifier()
    classifier.fit(features, Y)
    return classifier


def normalise_point(x):
    static, path = x
    path[:, 0] /= 2.
    return [static, path]


def normalise(X):
    return list(map(normalise_point, X))


def evaluate(classifier, features, Y):
    THRESHOLD = .3
    predictions_proba = classifier.predict_proba(features)[:, 1]
    predictions = [1. if pred >
                   THRESHOLD else 0. for pred in predictions_proba]
    cm = confusion_matrix(Y, predictions)
    Se = cm[1, 1] / float(cm[1, 1] + cm[1, 0])
    P = cm[1, 1] / float(cm[1, 1] + cm[0, 1])
    score = min(Se, P)
    print("Score of predictions: %s" % score)


def to_path(df, dynamic_variables):
    dim = len(dynamic_variables) + 1
    path = [[0.]*dim]
    for event in df.values:
        if event[1] in dynamic_variables:
            new_value = copy.deepcopy(path[-1])
            idx = 1 + dynamic_variables.index(event[1])
            new_value[idx] = event[2]
            hour, min = event[0].split(":")
            days = (float(hour) + float(min) / 60.)/24.
            new_value[0] = days
            path.append(new_value)

    path = np.array(path)
    unique_times = np.unique(path[:, 0])
    idx = []
    for time in unique_times:
        last_idx = np.where(path[:, 0] == time)[0][-1]
        idx.append(last_idx)
    path = path[idx]
    return path


def static_features(df, static_variables):
    return df[df["Parameter"].isin(static_variables)]["Value"].values


def reformat(X, static_variables, dynamic_variables):
    for i, x in enumerate(X):
        dynamic = to_path(x, dynamic_variables=dynamic_variables)
        static = static_features(x, static_variables=static_variables)
        X[i] = [static, dynamic]
    return X


def st2si(order, stream):
    if order > 1:
        return(tosig.stream2sig(stream, order))
    else:
        if order == 1:
            return np.concatenate((np.array([1]), stream[-1] - stream[0]), axis=0)
        else:
            return np.array([1])


def compute(X, order=2):
    func = partial(st2si, order)
    pool = Pool()
    n_samples = len(X)
    signatures = []
    try:
        signatures = np.array(list(tqdm(pool.imap(func, X), total=n_samples)))
    except Exception as e:
        print('Failed to compute signatures: ' + repr(e))
        signatures = []
    return signatures


def predict(classifier, url):
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=False)

    with open('data/test_input.txt', 'wb') as out:
        while True:
            data = r.read()
            if not data:
                break
            out.write(data)

    r.release_conn()

    data = {}
    df = pd.read_csv("data/test_input.txt")
    patient_id = int(df.values[0, 2])
    data[patient_id] = df

    X = []
    for patient_id in data:
        X.append(data[patient_id])

    X = reformat(X, static_variables=["Age", "Gender"], dynamic_variables=[
                 "Creatinine", "Glucose"])
    X = normalise(X)
    X = extract(X)

    # [0] means in-house dead [1] means in-house alive
    print('Predicted result: ' + classifier.predict(X))


if __name__ == "__main__":
    # DOWNLOAD & REFORMAT EVENT DATA, TRANSFORM TIME DEPENDENT VARIABLES
    X, Y = download()
    X = reformat(X, static_variables=["Age", "Gender"], dynamic_variables=[
                 "Creatinine", "Glucose"])

    # NORMALISE & EXTRACT FEATURES
    X = normalise(X)
    features = extract(X)

    # TRAIN THE MODEL BY SPLITING
    features_train, Y_train, features_test, Y_test = split(
        features, Y, proportion=0.75)
    classifier = train(features_train, Y_train)

    predict(classifier, 'https://storage.googleapis.com/challenge-2012-1.0.0.physionet.org/set-a/132539.txt')

    # EVALUATE PERFORMANCE
    evaluate(classifier, features_test, Y_test)
