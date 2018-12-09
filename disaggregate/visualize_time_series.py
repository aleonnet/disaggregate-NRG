from os.path import join, isfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from const import *


house_id = 1


def load_truth_dataframe(house_id):
    ''' load the truth dataframe from file.
    The main column is the aggregate building energy.
    '''
    # load truth data
    house_data = pd.read_csv(os.path.join(REDD_DIR, 'building_{0}.csv'.format(house_id)))
    house_data = house_data.set_index(pd.DatetimeIndex(house_data['time'])).drop('time', axis=1)

    devices = house_data.columns.values
    devices = devices[devices != 'main']

    return house_data, devices


def load_predicted_dataframe(house_id):
    ''' load prediction data.

    return a dataframe of predictions for each appliance.

    should have the same number of rows as the truth dataframe, and a column for
    each appliance like in the truth dataframe. the 'main' column in the truth,
    representing the aggregate building energy series, does not have to be present.

    '''
    #pred_data = pd.read_csv()

    return



def load_data(restriction='time'):
    ''' loads the truth and prediction dataframes and processes them lightly.
    '''

    truth_data, devices = load_truth_dataframe(house_id)

    pred_data = load_predicted_dataframe(house_id)


    # without real predictions I test on fake predictions
    fake_predictions = truth_data + truth_data * 0.3 + 20 * np.random.randn(*truth_data.shape, )
    pred_data = fake_predictions


    # subset to evaluation period
    eval_truth = truth_data[TRAIN_END:]
    eval_pred = pred_data[TRAIN_END:]


    if restriction == 'freq':
        # full time series is too long for memory: either downsample or plot a restricted set
        # downsample:
        eval_truth = eval_truth.resample("1H").mean()
        eval_pred = eval_pred.resample("1H").mean()

    elif restriction == 'time':
        # restricted time interval:
        END_PLOT_DATE = '2011-05-15'
        eval_truth = truth_data[TRAIN_END:END_PLOT_DATE]
        eval_pred = pred_data[TRAIN_END:END_PLOT_DATE]

    elif restriction is None:
        pass

    return eval_truth, eval_pred, devices


def ts_plot(eval_truth, eval_pred, devices):

    fig, ax = plt.subplots(len(devices) + 1, 1, sharex=True)

    ma = ax[0].plot_date(x=eval_truth.index.values, y=eval_truth['main'].values,
                            color='k', linewidth=1, fmt='-')
    ax[0].set_ylabel('Total energy')
    ax[0].set_title('Building {0}'.format(house_id))

    for i, app in enumerate(devices):
        tr, = ax[i+1].plot_date(x=eval_truth.index.values, y=eval_truth[app].values,
                                color='k', label='Truth', linewidth=2, fmt='-')
        pr, = ax[i+1].plot_date(x=eval_truth.index.values, y=eval_pred[app].values,
                                color='r', label='Predicted', linewidth=1, fmt='-')
        ax[i+1].set_ylabel(app)


    for j in range(len(devices) + 1):
        ax[j].spines['top'].set_visible(False)
        ax[j].spines['right'].set_visible(False)

    ax[0].legend(handles=[tr,pr])

    plt.show()



def piechart(eval_truth, eval_pred, devices):

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

    ###### Generates pie chart ######


    pie, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    patches, txt = ax1.pie(truth_val, autopct=None)
    ax1.axis('equal')
    ax1.set_title('True usage')

    patches, txt = ax2.pie(pred_val, autopct=None)
    ax2.axis('equal')
    ax2.set_title('Predicted usage')

    plt.legend(patches, truth_label, bbox_to_anchor=(1, 0.2), ncol=4)


    #draw circle
    centre_circle = plt.Circle((0,0),0.6,fc='white')
    ax1.add_artist(centre_circle)
    centre_circle = plt.Circle((0,0),0.6,fc='white')
    ax2.add_artist(centre_circle)


    # pie.set_size_inches(4.5, 8)
    # pie.savefig('pie1.png', dpi=300)
    plt.show()




if __name__ == '__main__':

    eval_truth, eval_pred, devices = load_data(restriction='time')
    # ts_plot(eval_truth, eval_pred, devices)

    eval_truth, eval_pred, devices = load_data(restriction=None)
    piechart(eval_truth, eval_pred, devices)
