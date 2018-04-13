import numpy as np
import pandas as pd
from load_data import loadData

class TableSummary():

    def __init__(self, dataset):

        self.dataset = dataset
        self.columns = self.dataset.columns
        self.digitalFeatures = []
        self.categroyFeatures = []
        self.timeFeature = []
        self.baiscInfo = {}
        self.recommendResult = {}

        self.getTimeFeature()
        self.columnsRecommend()
        self.getBasicInformation()

    def getBasicInformation(self):

        rowNum = len(self.dataset)
        colNum = self.dataset.values.shape[1]

        for feature in self.columns:
            try:
                self.dataset[feature] = pd.to_datetime(self.dataset[feature]) # 这里默认只有一个维度是time feature
                self.timeFeature = feature
                break
            except:
                continue

        if self.timeFeature == None:
            return -1

        sortTime = self.dataset[self.timeFeature].sort_values()
        timesatmp = sortTime.unique() # 时间戳去重，防止多个序列交织在一起

        minTime = str(timesatmp[0])
        maxTime = str(timesatmp[-1])
        interval = int(timesatmp[1]-timesatmp[0])/1e9

        self.baiscInfo = {"row_num": rowNum, "col_num": colNum, "time_feature": self.timeFeature, "digital_feature_num": len(self.digitalFeatures),
                "category_feature_num": len(self.categroyFeatures), "max_datetime": maxTime, "min_datetime": minTime, "interval": interval}

        #return self.baiscInfo

    def columnsRecommend(self):

        for feature in self.columns:
            if feature == self.timeFeature:
                continue
            elif self.dataset[feature].dtype == object:
                self.categroyFeatures.append(feature)
            elif "id" in feature.lower() or  "type" in feature.lower() or "code" in feature.lower(): # 这里只是简单rule是匹配，后续要精细化
                self.categroyFeatures.append(feature)
            else:
                self.digitalFeatures.append(feature)

        self.recommendResult = {"category_features": self.categroyFeatures, "digital_features": self.digitalFeatures, "time_feature": [self.timeFeature]}

    def getTimeFeature(self):

        for feature in self.columns:
            try:
                self.dataset[feature] = pd.to_datetime(self.dataset[feature]) # 这里默认只有一个维度是time feature
                self.timeFeature = feature
                break
            except:
                continue

if __name__ == '__main__':

    dataset = loadData('data/deepAD_data_summary_test_data1.csv')
    summary = TableSummary(dataset)
    print(summary.baiscInfo)
    print(summary.recommendResult)
















