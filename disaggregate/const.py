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


LEARN_TYPES = ['light','fridge','washer dryer','dish washer','electric oven',
                'sockets','air conditioning','electric furnace']
SAMPLE_PERIOD = 300
TRAIN_END = '2011-05-12'
