import pandas as pd
import numpy as np


def loadData(filename, choiceFeature=None):

    dataset = pd.read_csv(filename)
    if choiceFeature != None:
        dataset = dataset[choiceFeature]

    return dataset

def groupByScopeTS(dataset, choiceFeature):

    group = dataset.groupby(by=choiceFeature)
    series = {}
    for g in group:
        series[g[0]] = g[1]
    return series
