import pandas as pd
import numpy as np


def loadData(filename):

    dataset = pd.read_csv(filename)

    return dataset
