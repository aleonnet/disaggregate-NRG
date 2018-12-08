from os.path import join, isfile
import pandas as pd

from nilmtk.dataset_converters import convert_redd
from nilmtk import DataSet
from nilmtk import global_meter_group

from const import *


def show_top_devices(redd_data, n_top=5):
    ''' produce list of n_top power-using appliances for each building
    '''
    tops = []
    for i in range(1,7):
        tmp = redd_data.buildings[i]
        tops.append(tmp.elec.submeters().select_top_k(k=n_top))
    print(tops)


def subset_present_types(building, all_types):
    ''' search a specific building for appliance types present in the
    master list. returns a list of the found types.

    Input:
        building: a complete building instance

        all_types: a list of appliances to search for
    '''
    valid_list = []
    for item in all_types:
        try:
            building.elec.select_using_appliances(type=item)
            valid_list.append(item)
        except KeyError:
            continue

    return valid_list


def main():

    if not isfile(REDD_FILE):
        # convert raw data into hd5 file
        convert_redd(join(DATA_DIR, 'redd/low_freq'), REDD_FILE)

    redd = DataSet(REDD_FILE)

    for id in range(1,7):
        # parse all building data and generate dataframe
        meter_dict = {}

        # add aggregated building mains
        mains = redd.buildings[id].elec.mains().load(sample_period=SAMPLE_PERIOD)
        meter_dict['main'] = next(mains)

        # add building submeters
        present_list = subset_present_types(redd.buildings[id], LEARN_TYPES)
        for item in present_list:
            item_gen = redd.buildings[id].elec[item].load(sample_period=SAMPLE_PERIOD)
            meter_dict[item] = next(item_gen)

        # format dataframe
        tmp = pd.concat(meter_dict, axis=1)
        tmp.columns = tmp.columns.droplevel(level=(1,2))
        tmp = tmp[1:]       # remove NaN first row because of timestep aggregation

        tmp.to_csv(join(REDD_DIR,
                    'building_{0}.csv'.format(id)), index_label='time')


if __name__ == '__main__':
    main()
