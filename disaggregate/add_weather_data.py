import pandas as pd
import numpy as np

from const import SAMPLE_PERIOD


def load_weather():
    weather = pd.read_csv('../data/boston_hourly_weather.csv')
    weather = weather.set_index(pd.DatetimeIndex(weather['DATE'])).drop('DATE', axis=1)

    temp = weather['HOURLYDRYBULBTEMPF']
    precip = weather['HOURLYPrecip']
    pressure = weather['HOURLYStationPressure']

    df = pd.concat([temp, precip, pressure], axis=1)
    df.columns = ['temp','precip','pressure']

    # columns object to numeric
    for col in df:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # patch in missing values
    df = df.interpolate(method='time', axis=0)

    return df


def align_to_power_data(weather, power):

    weather = weather.resample('{0}s'.format(SAMPLE_PERIOD)).mean().interpolate(method='time', axis=0)
    merged_data = pd.merge(weather, power, left_index=True, right_index=True)

    return merged_data


def add_weather_to_power_data(power):
    weather = load_weather()
    merged_data = align_to_power_data(weather, power)

    return merged_data
