import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from visualize_time_series import ts_plot, piechart
from const import REDD_DIR, TRAIN_END


class RegressionModeler(object):

    def __init__(self, house_id, AR_terms):
        self.house_id = house_id
        self.AR_terms = AR_terms
        self.X_train = None
        self.X_test = None
        self.train_targets = None
        self.test_targets = None
        self.apps = None
        self.index = None
        self.truth = None

    def prepare_train_test_sets(self, targ):
        AR_terms = self.AR_terms

        house_data = pd.read_csv(os.path.join(REDD_DIR, 'building_{0}.csv'.format(self.house_id)))
        house_data = house_data.set_index(pd.DatetimeIndex(house_data['time'])).drop('time', axis=1)



        if targ == 'aggregate':
            # print(house_data.head())
            house_data = house_data.drop('Main', axis=1)
            house_data['Main'] = np.sum(house_data.values, axis=1)
            # print(house_data.head())
        elif targ == 'mains':
            pass

        apps = house_data.columns.values
        apps = apps[apps != 'Main']

        train_data = house_data[:TRAIN_END]
        test_data = house_data[TRAIN_END:]

        # construct X_train predictor matrix using autoregressive terms
        ar_list = []
        for i in range(AR_terms + 1):
            ar_list.append(train_data.Main.shift(i))

        X_train = pd.concat(ar_list, axis=1)
        X_train.columns = ['Main'] + ['AR{0}'.format(x) for x in range(1, AR_terms+1)]
        X_train = X_train[AR_terms:]

        # construct X_test predictor matrix using autoregressive terms
        ar_list = []
        for i in range(AR_terms + 1):
            ar_list.append(test_data.Main.shift(i))

        X_test = pd.concat(ar_list, axis=1)
        X_test.columns = ['Main'] + ['AR{0}'.format(x) for x in range(1, AR_terms+1)]
        X_test = X_test[AR_terms:]

        self.index = test_data.index[AR_terms:]
        self.truth = test_data.iloc[AR_terms:,:]

        # construct target variables. Because of autoregression 'cost', must throw
        # out AR_terms rows of the data
        train_targets = {}
        test_targets = {}
        for item in apps:
            train_targets[item] = train_data[item][AR_terms:]
            test_targets[item] = test_data[item][AR_terms:]

        self.X_train = X_train
        self.X_test = X_test
        self.train_targets = train_targets
        self.test_targets = test_targets
        self.apps = apps

    def fit_model(self, model):

        ### Prediction ###
        app_scores = []
        total_preds = []
        for target_app in self.apps:

            y_train = self.train_targets[target_app].values
            y_test = self.test_targets[target_app].values

            model.fit(self.X_train, y_train)

            preds = model.predict(self.X_test)
            app_scores.append(rmse(preds, y_test))
            total_preds.append(preds.reshape(-1,1))

        return app_scores, np.hstack(total_preds)

    def fit_multitask_model(self, model):

        y_train = np.hstack([self.train_targets[app].values.reshape(-1,1) for app in self.apps])
        y_test = np.hstack([self.test_targets[app].values.reshape(-1,1) for app in self.apps])

        model.fit(self.X_train, y_train)
        preds = model.predict(self.X_test)

        app_scores = []
        for i in range(len(self.apps)):
            app_scores.append(rmse(preds[:,i], y_test[:,i]))

        return app_scores, preds



def rmse(pred, target):
    return np.sqrt(np.mean((pred - target)**2))


def visuals(preds, index, apps, truth):
    output = pd.DataFrame(preds, index=index)
    output.columns = apps

    f = ts_plot(truth, output, apps)
    plt.show()

    f2 = piechart(truth, output, apps)
    plt.show()


def main():
    house_id = 1
    AR_terms = 48

    rmd = RegressionModeler(house_id, AR_terms)
    rmd.prepare_train_test_sets(targ='aggregate')


    # model = LinearRegression()
    # _, preds = rmd.fit_model(model)
    # visuals(preds, rmd.index, rmd.apps, rmd.truth)
    #
    # model = ElasticNetCV()
    # _, preds = rmd.fit_model(model)
    # visuals(preds, rmd.index, rmd.apps, rmd.truth)
    #
    #
    # model = RandomForestRegressor()
    # _, preds = rmd.fit_model(model)
    # visuals(preds, rmd.index, rmd.apps, rmd.truth)


    # # model = SVR()
    # # _, preds = rmd.fit_model(model)
    # #
    model = MultiTaskElasticNetCV()
    _, preds = rmd.fit_multitask_model(model)
    visuals(preds, rmd.index, rmd.apps, rmd.truth)


    # print(output.head(20))


if __name__ == '__main__':
    main()
