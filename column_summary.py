import numpy as np
import pandas as pd
from load_data import loadData
from base_summary import *
from sklearn.preprocessing import LabelEncoder


class ColumnsSummary(Summary):

    def __init__(self, dataset):

        super().__init__(dataset)

    def getBasicInformation(self):

        basicInfo = {}

        for feature in self.columns:

            min = "-"
            max = "-"
            median = "-"
            mean = "-"
            std = "-"

            if feature in self.digitalFeatures:
                min = int(np.min(self.dataset[feature]))
                max = int(np.max(self.dataset[feature]))
                median = int(np.median(self.dataset[feature]))
                mean = int(np.mean(self.dataset[feature]))
                std = int(np.std(self.dataset[feature]))

            count = int(len(self.dataset[feature]))
            uniqueNum = int(self.dataset[feature].nunique())
            nullNum = int(self.dataset[feature].isnull().sum())

            basicInfo[feature] = {"count": count, "min": min, "max": max, "median":median, "mean": mean,
                                       "std": std, "unique_num": uniqueNum, "null_num":nullNum}

        return basicInfo

    def getHist(self):

        histDict = {}

        for feature in self.columns:
            if feature is not self.timeFeature and feature in self.categroyFeatures:
                self.dataset[feature] = self.dataset[feature].fillna("other")
                transformer = LabelEncoder()
                self.dataset[feature] = transformer.fit_transform(self.dataset[feature])
                hist, binEdges = np.histogram(self.dataset[feature], bins=len(transformer.classes_))
                histDict[feature] = {"hist": hist.tolist(), "bin_edges": binEdges.tolist()}
        return histDict



if __name__ == '__main__':

    dataset = loadData('data/deepAD_data_summary_test_data1.csv')
    summary = ColumnsSummary(dataset)
    basicInfo = summary.getBasicInformation()
    hist = summary.getHist()
    print(basicInfo)
    print(hist)
