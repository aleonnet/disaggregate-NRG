import time
from os.path import join, isfile
import pandas as pd
import numpy as np
from six import iteritems
import matplotlib.pyplot as plt

from nilmtk.disaggregate import CombinatorialOptimisation, FHMM
from nilmtk import DataSet
from nilmtk import global_meter_group

from const import *
from utils import number_list_duplicates
from visualize_time_series import ts_plot, piechart



house_id = 6
def benchmarks(house_id):

    redd_train = DataSet(REDD_FILE)
    redd_test = DataSet(REDD_FILE)

    # set up training and test sets
    redd_train.set_window(end=TRAIN_END)
    redd_test.set_window(start=TRAIN_END)

    # get top N_DEV devices
    house = redd_train.buildings[house_id]
    test_elec = redd_test.buildings[house_id].elec
    top_apps = house.elec.submeters().select_top_k(k=N_DEV)

    # store mains data
    test_mains = next(test_elec.mains().load())


    truth = {}
    predictions = {}

    # benchmark classifier 1
    co = CombinatorialOptimisation()

    start = time.time()
    print("*" *20)
    print('Combinatorial Optimisation: ')
    print("*" *20)

    co.train(top_apps, sample_period=SAMPLE_PERIOD)
    truth['CO'], predictions['CO'] = predict(co, test_elec, SAMPLE_PERIOD, redd_train.metadata['timezone'])
    end = time.time()
    print("Runtime: ", end-start)


    # benchmark classifier 2
    fhmm = FHMM()

    start = time.time()
    print("*" *20)
    print('Factorial Hidden Markov Model: ')
    print("*" *20)

    fhmm.train(top_apps, sample_period=SAMPLE_PERIOD)
    truth['FHMM'], predictions['FHMM'] = predict(fhmm, test_elec, SAMPLE_PERIOD, redd_train.metadata['timezone'])

    end = time.time()
    print("Runtime: ", end-start)


    # add mains to truth
    truth['CO']['Main'] = test_mains
    truth['FHMM']['Main'] = test_mains

    return truth, predictions



def visualize_benchmarks(eval_truth, eval_pred, restriction='time'):
    if restriction == 'freq':
        # full time series is too long for memory: either downsample or plot a restricted set
        # downsample:
        eval_truth = eval_truth.resample("1H").mean()
        eval_pred = eval_pred.resample("1H").mean()

    elif restriction == 'time':
        # restricted time interval:
        END_PLOT_DATE = '2011-05-15'
        eval_truth = eval_truth[TRAIN_END:END_PLOT_DATE]
        eval_pred = eval_pred[TRAIN_END:END_PLOT_DATE]

    elif restriction is None:
        pass


    devices=eval_pred.columns.values

    f1 = piechart(eval_truth, eval_pred, devices=eval_pred.columns.values)
    plt.show()
    f2 = ts_plot(eval_truth, eval_pred, devices=eval_pred.columns.values)
    plt.show()

    # def compute_rmse(gt, pred):
    #     from sklearn.metrics import mean_squared_error
    #     rms_error = {}
    #     for appliance in gt.columns:
    #         rms_error[appliance] = np.sqrt(mean_squared_error(gt[appliance], pred[appliance]))
    #     return pd.Series(rms_error)
    #
    # rmse = {}
    # for clf_name in classifiers.keys():
    #     rmse[clf_name] = compute_rmse(gt, predictions[clf_name])
    # rmse = pd.DataFrame(rmse)
    #
    # print(rmse)


def predict(clf, test_elec, sample_period, timezone):
    pred = {}
    gt = {}

    for i, chunk in enumerate(test_elec.mains().load(sample_period=sample_period)):
        chunk_drop_na = chunk.dropna()
        pred[i] = clf.disaggregate_chunk(chunk_drop_na)
        gt[i]={}

        for meter in test_elec.submeters().meters:
            # Only use the meters that we trained on (this saves time!)
            gt[i][meter] = next(meter.load(sample_period=sample_period))
        gt[i] = pd.DataFrame({k:v.squeeze() for k,v in iteritems(gt[i])}, index=next(iter(gt[i].values())).index).dropna()

    # If everything can fit in memory
    gt_overall = pd.concat(gt)
    gt_overall.index = gt_overall.index.droplevel()
    pred_overall = pd.concat(pred)
    pred_overall.index = pred_overall.index.droplevel()

    # Having the same order of columns
    gt_overall = gt_overall[pred_overall.columns]

    #Intersection of index
    gt_index_utc = gt_overall.index.tz_convert("UTC")
    pred_index_utc = pred_overall.index.tz_convert("UTC")
    common_index_utc = gt_index_utc.intersection(pred_index_utc)


    common_index_local = common_index_utc.tz_convert(timezone)
    gt_overall = gt_overall.ix[common_index_local]
    pred_overall = pred_overall.ix[common_index_local]

    appliance_labels = [m.label() for m in gt_overall.columns.values]
    appliance_labels = number_list_duplicates(appliance_labels)

    gt_overall.columns = appliance_labels
    pred_overall.columns = appliance_labels
    return gt_overall, pred_overall


def main():

    for house_id in range(1,7):
        alltruth, allpreds = benchmarks(house_id)

        visualize_benchmarks(alltruth['CO'], allpreds['CO'], restriction='time')
        visualize_benchmarks(alltruth['FHMM'], allpreds['FHMM'], restriction='time')


if __name__ == '__main__':
    main()
