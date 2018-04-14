import numpy as np
import pandas as pd
from load_data import loadData
from base_summary import *

class TableSummary(Summary):

    def __init__(self, dataset):

        super().__init__(dataset)

    def getBasicInformation(self):

        if self.timeFeature == None:
            return {"Error": "No time feature"}

        rowNum = len(self.dataset)
        colNum = self.dataset.values.shape[1]

        sortTime = self.dataset[self.timeFeature].sort_values()
        timesatmp = sortTime.unique() # 时间戳去重，防止多个序列交织在一起

        minTime = str(timesatmp[0])
        maxTime = str(timesatmp[-1])
        interval = int(timesatmp[1]-timesatmp[0])/1e9

        basicInfo = {"row_num": rowNum, "col_num": colNum, "time_feature": self.timeFeature, "digital_feature_num": len(self.digitalFeatures),
                "category_feature_num": len(self.categroyFeatures), "max_datetime": maxTime, "min_datetime": minTime, "interval": interval}

        return basicInfo


if __name__ == '__main__':

    dataset = loadData('data/deepAD_data_summary_test_data1.csv')
    summary = TableSummary(dataset)
    basicInfo = summary.getBasicInformation()
    print(basicInfo)
    recommend = summary.getColumnRecommend()
    print(recommend)
















