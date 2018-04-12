import pandas as pd
import numpy as np


def loadTimeSeries(filename):

    ts = pd.read_csv(filename)

    return ts
