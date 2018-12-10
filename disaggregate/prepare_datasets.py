from os.path import join, isfile
import pandas as pd
from six import iteritems

from nilmtk.dataset_converters import convert_redd
from nilmtk import DataSet
from nilmtk import global_meter_group

from const import *
from utils import number_list_duplicates


def main():

    if not isfile(REDD_FILE):
        # convert raw data into hd5 file
        convert_redd(join(DATA_DIR, 'redd/low_freq'), REDD_FILE)

    redd = DataSet(REDD_FILE)

    # iterate over each building
    for id in range(1,7):
        # parse all building data and generate dataframe
        elec = redd.buildings[id].elec
        mains = next(elec.mains().load(sample_period=SAMPLE_PERIOD))

        # iterate over meters and gather time series
        meter_dict = {}
        for i, chunk in enumerate(elec.mains().load(sample_period=SAMPLE_PERIOD)):
            chunk_drop_na = chunk.dropna()

            meter_dict[i]={}
            for meter in elec.submeters().select_top_k(k=N_DEV).meters:
                meter_dict[i][meter] = next(meter.load(sample_period=SAMPLE_PERIOD))
            meter_dict[i] = pd.DataFrame({k:v.squeeze() for k,v in iteritems(meter_dict[i])},
                                            index=next(iter(meter_dict[i].values())).index).dropna()

        # If everything can fit in memory
        truth_df = pd.concat(meter_dict)
        truth_df.index = truth_df.index.droplevel()

        # format dataframe
        appliance_labels = [m.label() for m in truth_df.columns.values]
        fixed_appliance_labels = number_list_duplicates(appliance_labels)
        truth_df.columns = fixed_appliance_labels
        truth_df['Main'] = mains

        truth_df.to_csv(join(REDD_DIR,
                    'building_{0}.csv'.format(id)), index_label='time')

        # # add aggregated building mains
        # mains = redd.buildings[id].elec.mains().load(sample_period=SAMPLE_PERIOD)
        # meter_dict['main'] = next(mains)
        #
        # # new method: just select top n_dev devices for each building
        # top_app = redd.buildings[id].elec.submeters().select_top_k(k=n_dev)
        # top_app
        # top_app.meters

        # original method: pre-specifying list of devices, has compatibility issues later on
        # # add building submeters
        # present_list = subset_present_types(redd.buildings[id], LEARN_TYPES)
        # for item in present_list:
        #     item_gen = redd.buildings[id].elec[item].load(sample_period=SAMPLE_PERIOD)
        #     meter_dict[item] = next(item_gen)


        # tmp = pd.concat(meter_dict, axis=1)
        # tmp.columns = tmp.columns.droplevel(level=(1,2))
        # tmp = tmp[1:]       # remove NaN first row because of timestep aggregation


if __name__ == '__main__':
    main()
