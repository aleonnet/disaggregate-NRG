import time
from os.path import join, isfile
import pandas as pd
import numpy as np

from nilmtk.dataset_converters import convert_redd
from nilmtk import DataSet
from nilmtk import global_meter_group

from const import *
from prepare_datasets import subset_present_types


# TODO: check if the TRAIN_END format in const ('2011-05-12') is compatible with nilmtk indexing
TRAIN_END = '05-12-2011'

redd_train = DataSet(REDD_FILE)
redd_test = DataSet(REDD_FILE)

redd_train.set_window(end=TRAIN_END)
redd_test.set_window(start=TRAIN_END)
#
#
#
# # set buildings #1-5 for development, leaving #6 for holdout
# dev_train_buildings = [redd_train.buildings[i] for i in range(1,6)]
# dev_test_buildings = [redd_test.buildings[i] for i in range(1,6)]
#
# eval_buildings = [redd.buildings[6]]
#

def predict(clf, test_elec, sample_period, timezone):
    pred = {}
    gt= {}

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
    gt_overall.columns = appliance_labels
    pred_overall.columns = appliance_labels
    return gt_overall, pred_overall


from six import iteritems

from nilmtk.disaggregate import CombinatorialOptimisation, FHMM

for id in range(1,2):

    house = redd_train.buildings[id]

    test_elec = redd_test.buildings[id].elec
    # present_list = subset_present_types(house, LEARN_TYPES)
    # top_app = house.elec.select_using_appliances(type=present_list)

    top_apps = house.elec.submeters().select_top_k(k=5)
    #
    # fhmm = fhmm_exact.FHMM()
    # fhmm.train(top_apps, sample_period=SAMPLE_PERIOD)


    classifiers = {'CO':CombinatorialOptimisation(), 'FHMM':FHMM()}
    predictions = {}
    sample_period = 120
    for clf_name, clf in classifiers.items():

        start = time.time()

        print("*"*20)
        print(clf_name)
        print("*" *20)
        clf.train(top_apps, sample_period=sample_period)
        gt, predictions[clf_name] = predict(clf, test_elec, 120, redd_train.metadata['timezone'])

        end = time.time()
        print("Runtime: ", end-start)


def compute_rmse(gt, pred):
    from sklearn.metrics import mean_squared_error
    rms_error = {}
    for appliance in gt.columns:
        rms_error[appliance] = np.sqrt(mean_squared_error(gt[appliance], pred[appliance]))
    return pd.Series(rms_error)

rmse = {}
for clf_name in classifiers.keys():
    rmse[clf_name] = compute_rmse(gt, predictions[clf_name])
rmse = pd.DataFrame(rmse)

print(rmse)
