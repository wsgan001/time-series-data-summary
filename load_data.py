import pandas as pd
import numpy as np


def loadTimeSeries(filename):

    ts = pd.read_csv(filename)
    tsValue = ts["value"]

    return ts, tsValue
