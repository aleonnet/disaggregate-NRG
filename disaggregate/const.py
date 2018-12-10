''' User/OS specific constants to enable running the same code from
different platforms '''

import sys
import os

# fred's configuration
if sys.platform == 'linux':
    DATA_DIR = '/mnt/e/data'
    REDD_DIR = os.path.join(DATA_DIR, 'redd')
    REDD_FILE = os.path.join(REDD_DIR, 'redd.h5')

elif sys.platform == 'win32':
    pass

elif sys.platform == 'darwin':
    pass


# number of top devices to extract for each building in prepare_datasets.py
N_DEV = 6

# time series aggregation window (300 s = 5 min)
SAMPLE_PERIOD = 300

# cutoff day between training period and test period
TRAIN_END = '2011-05-12'
