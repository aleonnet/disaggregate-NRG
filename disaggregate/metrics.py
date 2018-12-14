import numpy as np
import pandas as pd
import math


def fraction_energy_assigned_correctly(pred, truth):
    '''Compute fraction of energy assigned correctly

    Returns fraction : float in the range [0,1]
    '''

    # drop main column if it exists
    try:
        eval_truth = truth.drop('Main', axis=1)
    except ValueError:
        eval_truth = truth
    try:
        eval_pred = pred.drop('Main', axis=1)
    except ValueError:
        eval_pred = pred


    apps = eval_truth.columns.values
    assert set(apps) == set(eval_pred.columns.values), "appliances don't match!"

    total_predicted_energy = np.sum(eval_pred.values)
    total_true_energy = np.sum(eval_truth.values)

    frac = 0
    for app in apps:
        truth_frac = eval_truth[app].sum() / total_true_energy
        pred_frac = eval_pred[app].sum() / total_predicted_energy
        frac += min(truth_frac, pred_frac)

    return frac


def average_normalized_appliance_mae(pred, truth):
    ''' average of the scaled mean absolute error of each appliance
    over the time period (scaled by the average true signal of each appliance).

    Returns error >= 0
    '''

    # drop main column if it exists
    try:
        eval_truth = truth.drop('Main', axis=1)
    except ValueError:
        eval_truth = truth
    try:
        eval_pred = pred.drop('Main', axis=1)
    except ValueError:
        eval_pred = pred


    apps = eval_truth.columns.values
    assert set(apps) == set(eval_pred.columns.values), "appliances don't match!"

    total_predicted_energy = np.sum(eval_pred.values)
    total_true_energy = np.sum(eval_truth.values)

    errors = []
    for app in apps:
        ae = np.abs(eval_truth[app] - eval_pred[app])
        errors.append(np.mean(ae) / eval_truth[app].mean())

    return np.mean(errors)
