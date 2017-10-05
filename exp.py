from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, ensemble
import xgboost as xgb
import lightgbm as lgb

"""
I'm not a programmer, I don't even play one on TV. These are bits and pieces of work by people a lot smarter than I am,
assembled here for our collective use to experiment with.

Xander Dunn - Initial structure
the1Owl (Kaggle) - model "for" statement
ObjectScience - hacked the "for" statement to include lgb and integrated Xander's code.
Jim Flemming - Ensemble

This is an early stage framework to help you get up and running with your model selection and parameter tuning.
Hopefully it gets you past the some of the design stage and into the Feature Engineering stage quickly.
I'm sure there is plenty here that can be improved on or I've missed and I'm happy to make changes if I've botched something up.

Developed and tested on Win 10 and Py3.X
XGBoost and LightGBM aren't trivial installs if you're new and on Windows (but will probably get easier as time marches forward)
If you have trouble, keep digging, eveyrthing you need is "out there" you just have to grind through to find it.

"""


def main():
    # Set seed for reproducibility
    np.random.seed(2017)

    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv('train_data.csv', header=0)
    prediction_data = pd.read_csv('predict_data.csv', header=0)

    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(training_data) if "feature" in f]
    X = training_data[
        features]  # some things like tensorflow require you actually declare as 'X = np.array(training_data[features])', at least it did for me.
    Y = training_data["target"]
    x_prediction = prediction_data[features]
    ids = prediction_data["id"]

    # feature engineering
    # ================================================
    # This is all you... get creative. Good Luck!
    # ================================================

    # Parameters
    g = {'ne': 100, 'md': 4,
         'rs': 2016}  # If you can get max_features to work (see below) you can define it here as - 'mf':50, - g={'ne':100,'md':4, 'mf':50,'rs':2016}

    # Models | there is a bug here somewhere that if you call 'max_features' it throws an error. For now leave it out (default)
    etc = ensemble.ExtraTreesClassifier(n_estimators=g['ne'], max_depth=g['md'], random_state=g['rs'],
                                        criterion='entropy', min_samples_split=4, min_samples_leaf=2, verbose=0,
                                        n_jobs=12)
    etr = ensemble.ExtraTreesRegressor(n_estimators=g['ne'], max_depth=g['md'], random_state=g['rs'],
                                       min_samples_split=4, min_samples_leaf=2, verbose=0, n_jobs=12)
    rfc = ensemble.RandomForestClassifier(n_estimators=g['ne'], max_depth=g['md'], random_state=g['rs'],
                                          criterion='entropy', min_samples_split=4, min_samples_leaf=2, verbose=0,
                                          n_jobs=12)
    rfr = ensemble.RandomForestRegressor(n_estimators=g['ne'], max_depth=g['md'], random_state=g['rs'],
                                         min_samples_split=4, min_samples_leaf=2, verbose=0, n_jobs=12)
    xgc = xgb.XGBClassifier(n_estimators=g['ne'], max_depth=g['md'], seed=g['rs'], missing=np.nan, learning_rate=0.02,
                            subsample=0.9, colsample_bytree=0.85, objective='binary:logistic')
    xgr = xgb.XGBRegressor(n_estimators=g['ne'], max_depth=g['md'], seed=g['rs'], missing=np.nan, learning_rate=0.02,
                           subsample=0.9, colsample_bytree=0.85, objective='reg:linear')
    lgc = lgb.LGBMClassifier(n_estimators=g['ne'], max_depth=g['md'], learning_rate=0.01, nthread=12)
    lgr = lgb.LGBMRegressor(n_estimators=g['ne'], max_depth=g['md'], learning_rate=0.01, nthread=12)

    # Designate model(s) to run (you can also throw some CV in here as well... get your learn on!)
    clf = {'etc': etc, 'etr': etr, 'rfc': rfc, 'rfr': rfr, 'xgr': xgr, 'xgc': xgc, 'lgr': lgr, 'lgc': lgc}
    # clf = {'':} # Use this if you want to run a single model or a subset of models, ex. {'etr':etr} or {'etc':etc, 'rfc':rfc, 'xgc':xgc, 'lgc':lgc}

    # Train and predict
    for c in clf:
        if c[:1] != "x" and c[:1] != "l":  # not xgboost or lightgbm
            model = clf[c]
            model.fit(X, Y)
            if c[-1] != "c":  # not classifier
                results = model.predict(x_prediction)
                results_df = pd.DataFrame(data={'probability': results})
                joined = pd.DataFrame(ids).join(results_df)
                csv_path = 'predictions_{}.csv'.format(c)
                joined.to_csv(csv_path, index=None)
            else:  # classifier
                results = model.predict_proba(x_prediction)[:, 1]
                results_df = pd.DataFrame(data={'probability': results})
                joined = pd.DataFrame(ids).join(results_df)
                csv_path = 'predictions_{}.csv'.format(c)
                joined.to_csv(csv_path, index=None)
        elif c[:1] == "x":  # xgboost
            model = clf[c]
            model.fit(X, Y)
            if c == "xgr":  # xgb regressor
                results = model.predict(x_prediction)
                results_df = pd.DataFrame(data={'probability': results})
                joined = pd.DataFrame(ids).join(results_df)
                csv_path = 'predictions_{}.csv'.format(c)
                joined.to_csv(csv_path, index=None)
            else:  # xgb classifier
                results = model.predict_proba(x_prediction)[:, 1]
                results_df = pd.DataFrame(data={'probability': results})
                joined = pd.DataFrame(ids).join(results_df)
                csv_path = 'predictions_{}.csv'.format(c)
                joined.to_csv(csv_path, index=None)
        else:  # lightgbm
            model = clf[c]
            model.fit(X, Y)
            if c == "lgr":  # lgb regressor
                results = model.predict(x_prediction)
                results_df = pd.DataFrame(data={'probability': results})
                joined = pd.DataFrame(ids).join(results_df)
                csv_path = 'predictions_{}.csv'.format(c)
                joined.to_csv(csv_path, index=None)
            else:  # lgb classifier
                results = model.predict_proba(x_prediction)[:, 1]
                results_df = pd.DataFrame(data={'probability': results})
                joined = pd.DataFrame(ids).join(results_df)
                csv_path = 'predictions_{}.csv'.format(c)
                joined.to_csv(csv_path, index=None)

    # Ensemble

    paths = [
        'predictions_etc.csv',
        'predictions_etr.csv',
        'predictions_rfc.csv',
        'predictions_rfc.csv',
        'predictions_xgc.csv',
        'predictions_xgr.csv',
        'predictions_lgc.csv',
        'predictions_lgr.csv'
    ]

    t_id = []
    probs = []
    for path in paths:
        df = pd.read_csv(path)
        t_id = df['id'].values
        probs.append(df['probability'].values)

    probability = np.power(np.prod(probs, axis=0), 1.0 / len(paths))
    assert (len(probability) == len(t_id))

    df_pred = pd.DataFrame({
        'id': t_id,
        'probability': probability,
    })
    csv_path = 'predictions_ensemble_{}.csv'.format(int(time.time()))
    df_pred.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))


if __name__ == '__main__':
    main()