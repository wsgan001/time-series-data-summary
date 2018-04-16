import pandas as pd
import numpy as np
import argparse
import config


def loadData(filename):

    dataset = pd.read_csv(filename)

    return dataset


def groupByScopeTS(dataset, choiceFeature):

    group = dataset.groupby(by=choiceFeature)
    series = {}
    for g in group:
        series[g[0]] = g[1]
    return series


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepAD data load')
    parser.add_argument('--filepath', '-fp', type=str, help='dataset csv file path')
    args = parser.parse_args()

    filePath = args.filepath
    dataset = loadData(filePath)
    config.setVar(dataset)
    print(config.getVar())







