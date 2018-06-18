
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics, preprocessing
from xgboost import XGBClassifier
import numpy as np

__author__ = 'Doto Damien Pandam <damiengroover@gmail.com>'


def main():
    np.random.seed(0)
    print('loading....')
    training_data = pd.read_csv('numerai_training_data.csv', header=0)
    predict_data = pd.read_csv('numerai_tournament_data.csv', header=0)
    validation = predict_data[predict_data['data_type']=='validation']

    train_bernie = training_data.drop([
        'id', 'era', 'data_type',
        'target_charles', 'target_elizabeth',
        'target_jordan', 'target_ken'], axis=1)

    features = [f for f in list(train_bernie) if "feature" in f]
    training_features = train_bernie[features]
    training_targets = train_bernie["target_bernie"]
    prediction_features = predict_data[features]
    x_prediction = validation[features]
    ids = predict_data["id"]

    xgc = XGBClassifier(nthread=-1, learning_rate=0.10, subsample=0.6, colsample_bytree=0.9, n_estimators=100,
                        max_depth=4, min_child_weight=3, silent=1, objective='binary:logistic')

    etc = ExtraTreesClassifier(criterion='entropy', verbose=0, n_jobs=-1, min_samples_split=2, min_samples_leaf=2,
                               n_estimators=100, max_features=2, max_depth=3, random_state=1)

    rfc = RandomForestClassifier(criterion='entropy', verbose=0, n_jobs=-1, min_samples_split=2, min_samples_leaf=2,
                                 n_estimators=100, max_features=2, max_depth=3, random_state=1)

    lr = LogisticRegression(C=100, solver='lbfgs',
                            penalty='l2', tol=1e-6, n_jobs=-1, max_iter=50, )

    adc = AdaBoostClassifier(
        base_estimator=etc, n_estimators=50, learning_rate=0.1, )

    sclf = StackingClassifier(classifiers=[xgc, rfc, lr], meta_classifier=adc, use_probas=True,
                              average_probas=False)


    print("Training...")
    model = sclf
    model.fit(training_features, training_targets)

    print("Predicting...")

    _predictions = model.predict_proba(prediction_features)[:, 1]
    y_predictions = model.predict_proba(x_prediction)[:, 1]

    print("- probabilities:", y_predictions[1:6])

    print("- target:", validation['target_bernie'][1:6])
    print("- rounded probability:", [round(p) for p in y_predictions][1:6])
    # accuracy
    correct = [round(x)==y for (x,y) in zip(y_predictions, validation['target_bernie'])]
    print("- accuracy: ", sum(correct)/float(validation.shape[0]))
     # Our validation logloss 
    print("- validation logloss:", metrics.log_loss(validation['target_bernie'], y_predictions))
    # format and save result in csv files
    _results = pd.DataFrame(data={'probability': _predictions})
    result = pd.DataFrame(ids).join(_results)

    print("Writing predictions")
    result.to_csv("bernie_predictions.csv", index=False)


if __name__ == '__main__':
    main()
