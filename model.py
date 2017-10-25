
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np

__author__ = 'Doto Damien Pandam <damiengroover@gmail.com>'


def classify():
    print('loading....')
    training_data = pd.read_csv('train_data.csv', header=0)
    predict_data = pd.read_csv('predict_data.csv', header=0)

    features = [f for f in list(training_data) if "feature" in f]
    training_features = training_data[features]
    training_targets = training_data["target"]
    prediction_features = predict_data[features]
    ids = predict_data["id"]

    xgc = XGBClassifier(nthread=-1, learning_rate=0.10, subsample=0.6, colsample_bytree=0.9, n_estimators=100,
                        max_depth=4, min_child_weight=3, silent=1, objective='binary:logistic')

    etc = ExtraTreesClassifier(criterion='entropy', verbose=0, n_jobs=-1, min_samples_split=2, min_samples_leaf=2,
                               n_estimators=100, max_features=2, max_depth=3, random_state=1)

    

    rfc = RandomForestClassifier(criterion='entropy', verbose=0, n_jobs=-1, min_samples_split=2, min_samples_leaf=2,
                                 n_estimators=100, max_features=2, max_depth=3, random_state=1)

    lr = LogisticRegression(C=100, solver='lbfgs', penalty='l2', tol=1e-6, n_jobs=-1, max_iter=50, )

    adc = AdaBoostClassifier(base_estimator=etc, n_estimators=50, learning_rate=0.1, )

    sclf = StackingClassifier(classifiers=[xgc, rfc, lr], meta_classifier=adc, use_probas=True,
                              average_probas=False)

    params = {'kneighborsclassifier__n_neighbors': [1, 5],
              'randomforestclassifier__n_estimators': [10, 50],
              'meta-logisticregression__C': [0.1, 10.0]}

    print("Training...")
    model = sclf
    model.fit(training_features, training_targets)

    print("Predicting...")
    _predictions = model.predict_proba(prediction_features)[:, 1]
    _results = pd.DataFrame(data={'probability': _predictions})
    result = pd.DataFrame(ids).join(_results)
    # accuracy = accuracy_score(_predictions, labels_test)

    print("Writing predictions")
    result.to_csv("predictions1.csv", index=False)


classify()
