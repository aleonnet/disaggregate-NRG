import numpy as np
import pandas as pd
import math


def fraction_energy_assigned_correctly(predictions, ground_truth):
    '''Compute fraction of energy assigned correctly

        fraction =
        \\sum_n min \\left (
        \\frac{\\sum_n y}{\\sum_{n,t} y},
        \\frac{\\sum_n \\hat{y}}{\\sum_{n,t} \\hat{y}}
        \\right )

    Returns fraction : float in the range [0,1]

    '''

    # drop main column if it exists
    try:
        eval_truth = eval_truth.drop('main', axis=1)
    except KeyError:
        pass
    try:
        eval_pred = eval_pred.drop('main', axis=1)
    except KeyError:
        pass

    truth_label = eval_truth.sum().index
    truth_val = eval_truth.sum().values

    pred_label = eval_pred.sum().index
    pred_val = eval_pred.sum().values

    # predictions_submeters = MeterGroup(meters=predictions.submeters().meters)
    # ground_truth_submeters = MeterGroup(meters=ground_truth.submeters().meters)
    #
    #
    # fraction_per_meter_predictions = predictions_submeters.fraction_per_meter()
    # fraction_per_meter_ground_truth = ground_truth_submeters.fraction_per_meter()
    #
    # fraction_per_meter_ground_truth.index = fraction_per_meter_ground_truth.index.map(lambda meter: meter.instance)
    # fraction_per_meter_predictions.index = fraction_per_meter_predictions.index.map(lambda meter: meter.instance)

    fraction = 0
    for meter_instance in predictions_submeters.instance():
        fraction += min(fraction_per_meter_ground_truth[meter_instance],
                        fraction_per_meter_predictions[meter_instance])
    return fraction
