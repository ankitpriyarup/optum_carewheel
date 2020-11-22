import os.path
import sys
import numpy as np
import pandas as pd
import sklearn
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from acumos.modeling import Model, List, create_namedtuple
from acumos.session import Requirements
from os import path
from image_mood_classifier._version import MODEL_NAME, __version__ as VERSION
from image_mood_classifier.prediction_formatter import Formatter
from image_mood_classifier._version import MODEL_NAME


def load_dataset(path_features=None):
    from image_mood_classifier.prediction_formatter import Formatter
    if path_features is None:
        dummySample = {Formatter.COL_NAME_IDX: 0,
                       Formatter.COL_NAME_CLASS: "toy", Formatter.COL_NAME_PREDICTION: 0.243}
        df = pd.DataFrame([dummySample])
        return df.drop([0])
    df = pd.read_csv(path_features)
    return df


def create_keras_model_(unit_size=32, input_size=8, label_size=1):
    classifier = Sequential()
    classifier.add(Dense(unit_size, activation='relu', input_dim=input_size))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(unit_size, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(label_size, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    classifier.compile(loss='categorical_crossentropy',
                       optimizer=sgd, metrics=['accuracy'])
    return classifier


def classifier_train(X, y, method="svc"):
    if method == "rf":
        classifier = RandomForestClassifier()
        param_grid = [
            {'n_estimators': [10, 100, 300, 500], 'min_samples_split': [2, 10]}
        ]
    elif method == "svc":
        classifier = SVC()
        param_grid = [
            {'kernel': ['rbf'], 'gamma': [
                1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
        ]
    elif method == "dnn":
        from keras.wrappers.scikit_learn import KerasClassifier
        from functools import partial  # partial function
        build_partial = partial(create_keras_model_,
                                input_size=X.shape[1], label_size=1)
        classifier = KerasClassifier(build_fn=build_partial, verbose=0)
        param_grid = [
            {'unit_size': [32, 64, 256, 512]},
        ]
    clf = GridSearchCV(classifier, param_grid, cv=5, n_jobs=-1, verbose=2)
    clf.fit(X, y)
    classifier = clf.best_estimator_
    print([classifier, clf.best_params_, clf.best_score_])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y)
    classifier.fit(X_train, y_train)
    y_true, y_pred = y_test, classifier.predict(X_test)
    print(classification_report(y_true, y_pred))
    classifier.fit(X, y)
    return classifier


def model_create_pipeline(formatter, clf):
    formatter.set_params(classifier=clf)
    tag_type = []
    for item in formatter.output_types_:
        for k in item:
            tag_type.append((k, item[k]))
    name_in = "ImageTag"
    ImageTag = create_namedtuple(name_in, tag_type)
    name_multiple_in = name_in + "s"
    ImageTagSet = create_namedtuple(
        name_in + "Set", [(name_multiple_in, List[ImageTag])])

    def predict_class(val_wrapped: ImageTagSet) -> ImageTagSet:
        df = pd.DataFrame(getattr(val_wrapped, name_multiple_in),
                          columns=ImageTag._fields)
        tags_df = formatter.predict(df)
        tags_parts = tags_df.to_dict('split')
        tags_list = [ImageTag(*r) for r in tags_parts['data']]
        print("[{} - {}:{}]: Input {} row(s) ({}), output {} row(s) ({}))".format(
              "classify", MODEL_NAME, VERSION, len(df), ImageTagSet, len(tags_df), ImageTagSet))
        return ImageTagSet(tags_list)
    package_path = path.dirname(path.realpath(__file__))
    return Model(classify=predict_class), Requirements(packages=[package_path], reqs=[pd, np, sklearn])


def model_archive(clf=None, debugging=False):
    if not debugging:
        return None
    import pickle
    if clf is None:
        if os.path.exists('model_cf.pkl'):
            print("DEBUG ARCHIVE: Loading an old model...")
            with open("model_cf.pkl", "rb") as f:
                clf = pickle.load(f)
    elif not os.path.exists('model_cf.pkl'):
        print("Saving a new model...")
        with open("DEBUG ARCHIVE: model_cf.pkl", "wb") as f:
            pickle.dump(clf, f)
    return clf


def main(config={}):
    parser = argparse.ArgumentParser()
    submain = parser.add_argument_group(
        'main execution and evaluation functionality')
    submain.add_argument('-p', '--predict_path', type=str, default='',
                         help="Save predictions from model (model must be provided via 'dump_model')")
    submain.add_argument('-i', '--input', type=str, default='',
                         help='Absolute path to input training data file. (for now must be a header-less CSV)')
    submain.add_argument('-C', '--cuda_env', type=str, default='',
                         help='Anything special to inject into CUDA_VISIBLE_DEVICES environment string')
    subopts = parser.add_argument_group(
        'model creation and configuration options')
    subopts.add_argument('-l', '--labels', type=str, default='',
                         help="Path to label one-column file with one row for each input")
    subopts.add_argument('-m', '--model_type', type=str, default='rf',
                         help='specify the underlying classifier type (rf (randomforest), svc (SVM))', choices=['svm', 'rf'])
    subopts.add_argument('-f', '--feature_nomask', dest='feature_nomask',
                         default=False, action='store_true', help='create masked samples on input')
    subopts.add_argument('-n', '--add_softnoise', dest='softnoise', default=False,
                         action='store_true', help='do not add soft noise to classification inputs')
    subopts.add_argument('-a', '--push_address',
                         help='server address to push the model (e.g. http://localhost:8887/upload)', default=os.getenv('ACUMOS_PUSH', ""))
    subopts.add_argument('-A', '--auth_address',
                         help='server address for login and push of the model (e.g. http://localhost:8887/auth)', default=os.getenv('ACUMOS_AUTH', ""))
    subopts.add_argument(
        '-d', '--dump_model', help='dump model to a pickle directory for local running', default='')
    subopts.add_argument('-s', '--summary', type=int, dest='summary', default=0,
                         help='summarize top N image classes are strong for which label class (only in training)')
    config.update(vars(parser.parse_args()))
    if not os.path.exists(config['input']):
        print("The target input '{:}' was not found, please check input arguments.".format(
            config['input']))
        sys.exit(-1)
    print("Loading raw samples...")
    rawDf = pd.read_csv(config['input'], delimiter=",")
    if config['cuda_env']:
        os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_env']
    if not config['predict_path'] and config['labels']:
        if not os.path.exists(config['labels']):
            print("The target labels '{:}' was not found, please check input arguments.".format(
                config['labels']))
            sys.exit(-1)
        formatter = Formatter(input_softnoise=config['softnoise'])
        print("Loading labels to train a new model...")
        rawLabel = pd.read_csv(config['labels'], header=None, delimiter=",")
        if len(rawLabel.columns) != 1:
            print(
                "Error, not currently programmed to best-of class selection to a singleton.")
            sys.exit(-1)
        rawLabel = rawLabel[0].tolist()
        formatter.learn_input_mapping(rawDf, "tag", "image", "score")
        print("Converting block of {:} responses into training data, utilizing {:} images...".format(
            len(rawDf), len(rawLabel)))
        objRefactor = formatter.transform_raw_sample(
            rawDf, rawLabel, None if config['feature_nomask'] else Formatter.SAMPLE_GENERATE_MASKING)
        print("Generated {:} total samples (skip-masking: {:})".format(
            len(objRefactor['values']), config['feature_nomask']))
        clf = model_archive()
        if config['summary']:
            df_combined = pd.DataFrame(
                objRefactor['values'], columns=objRefactor['columns'])
            df_combined['_labels'] = objRefactor['labels']
            groupSet = df_combined.groupby('_labels')
            for nameG, rowsG in groupSet:
                df_sum = rowsG.sum(axis=0, numeric_only=True)
                series_top = df_sum.sort_values(ascending=False)
                print("Label: '{:}', top {:} classes...".format(
                    nameG, config['summary']))
                print(series_top[0:config['summary']])
        if config['push_address'] or config['dump_model']:
            if clf is None:
                clf = classifier_train(
                    objRefactor['values'], objRefactor['labels'], config['model_type'])
            model, reqs = model_create_pipeline(formatter, clf)
            model_archive(clf)
            if config['push_address']:
                from acumos.session import AcumosSession
                session = AcumosSession(
                    push_api=config['push_address'], auth_api=config['auth_address'])
                session.push(model, MODEL_NAME, reqs)
                print("Pushing new model to '{:}'...".format(
                    config['push_address']))
            if config['dump_model']:
                from acumos.session import AcumosSession
                from os import makedirs
                if not os.path.exists(config['dump_model']):
                    makedirs(config['dump_model'])
                print("Dumping new model to '{:}'...".format(
                    config['dump_model']))
                session = AcumosSession()
                session.dump(model, MODEL_NAME, config['dump_model'], reqs)
    else:
        if not config['dump_model'] or not os.path.exists(config['dump_model']):
            print("Attempting to predict from a dumped model, but model not found.".format(
                config['dump_model']))
            sys.exit(-1)

        print("Attempting predict/transform on input sample...")
        from acumos.wrapped import load_model
        model = load_model(config['dump_model'])

        type_in = model.classify._input_type
        classify_in = type_in(*tuple(col for col in rawDf.values.T))
        out_wrapped = model.classify.from_wrapped(classify_in).as_wrapped()
        dfPred = pd.DataFrame(out_wrapped[0])
        if config['predict_path']:
            print("Writing prediction to file '{:}'...".format(
                config['predict_path']))
            dfPred.to_csv(config['predict_path'], sep=",", index=False)
        if dfPred is not None:
            print("Predictions:\n{:}".format(dfPred))


if __name__ == '__main__':
    pathRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pathRoot not in sys.path:
        sys.path.append(pathRoot)
    main()
