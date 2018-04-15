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
                min = float(np.min(self.dataset[feature]))
                max = float(np.max(self.dataset[feature]))
                median = float(np.median(self.dataset[feature]))
                mean = float(np.mean(self.dataset[feature]))
                std = float(np.std(self.dataset[feature]))

            count = int(len(self.dataset[feature]))
            uniqueNum = int(self.dataset[feature].nunique())
            nullNum = int(self.dataset[feature].isnull().sum())

            basicInfo[feature] = {"count": count, "min": min, "max": max, "median": median, "mean": mean,
                                  "std": std, "unique_num": uniqueNum, "null_num": nullNum}

        return basicInfo

    def getHist(self):

        histDict = {}

        for feature in self.columns:

            if feature is not self.timeFeature and feature in self.categroyFeatures:

                self.dataset[feature] = self.dataset[feature].fillna("null")
                group = self.dataset.groupby(by=feature)
                d = {}
                for g in group:
                    d[g[0]] = len(g[1])
                histDict[feature] = d

            elif feature in self.digitalFeatures:
                hist, binEdges = np.histogram(self.dataset[feature], bins="auto")
                histDict[feature] = {"hist": hist.tolist(), "bin_edges": binEdges.tolist()}

        return histDict

    def getRank(self, top=10, factor=2): # 取前10个值，判断因此为最大的超过第二大的factor倍

        dominDict = {}

        for feature in self.categroyFeatures:

            self.dataset[feature] = self.dataset[feature].fillna("null")
            group = self.dataset.groupby(by=feature)
            d = {}
            for g in group:
                d[g[0]] = len(g[1])

            sort_d = sorted(d.items(), key=lambda x: x[1], reverse=True)

            if len(sort_d) > 1:
                if sort_d[0][1] > factor*sort_d[1][1]:
                    dominDict[feature] = dict(sort_d[:top])
            else:
                dominDict[feature] = dict(sort_d[:top])

        return dominDict

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    dataset = loadData('data/deepAD_data_summary_test_data1.csv')
    summary = ColumnsSummary(dataset)
    basicInfo = summary.getBasicInformation()
    hist = summary.getHist()
    print(basicInfo)
    print(hist)

    dominDict = summary.getRank()
    print(dominDict)
    for key in dominDict.keys():
        domainInfo = dominDict[key]
        plt.bar(range(len(domainInfo)), domainInfo.values())
        plt.show()

    # import matplotlib.pyplot as plt
    #
    # for feature in summary.columns:
    #     # if feature in summary.categroyFeatures:
    #     #     plt.bar(range(len(hist[feature])), hist[feature].values())
    #     #     plt.show()
    #     if feature in summary.digitalFeatures:
    #         print(hist[feature]['bin_edges'])
    #         print(hist[feature]["hist"])
    #         plt.bar(hist[feature]['bin_edges'][:-1], hist[feature]["hist"])
    #         plt.show()

