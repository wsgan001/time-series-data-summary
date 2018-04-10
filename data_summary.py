import pandas as pd
import numpy as np
from load_data import *
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose


class StatisticInfo():

    def __init__(self):
        self.num = None
        self.mean = None
        self.std = None
        self.minNum = None
        self.maxNum = None
        self.median = None
        self.uniqueNum = None
        self.nullNum = None

    def getInfo(self):

        return {"num":self.num, "mean":self.mean, "std":self.std, "min":self.minNum,
                "max":self.maxNum, "median":self.median, "unique_num": self.uniqueNum, "null_num":self.nullNum}

class DataSummary():

    def __init__(self):

        self.ts = None
        self.tsValue = None

    def loadData(self, filename):

        ts, tsValue = loadTimeSeries(filename)
        self.ts = ts
        self.tsValue = tsValue

    def getStatisticInfo(self):

        statisticInfo = StatisticInfo()

        statisticInfo.num = len(self.tsValue)
        statisticInfo.mean = np.mean(self.tsValue)
        statisticInfo.std = np.std(self.tsValue)
        statisticInfo.minNum = np.min(self.tsValue)
        statisticInfo.maxNum = np.max(self.tsValue)
        statisticInfo.median = np.median(self.tsValue)
        statisticInfo.uniqueNum = self.tsValue.nunique()
        statisticInfo.nullNum = pd.isnull(self.tsValue).sum()

        return statisticInfo

    def stationarity(self):

        ADF_p_value = adfuller(self.tsValue, autolag='AIC')[1]
        return ADF_p_value

    def randomness(self, lags):

        p_value = acorr_ljungbox(self.tsValue, lags=lags)[1]
        return p_value

    def diff(self, order):

        self.tsValue = np.diff(self.tsValue, n=order)

    def decomposition(self, freq):

        decomposition = seasonal_decompose(self.tsValue, model="additive", )  # 季节分解
        return decomposition.trend, decomposition.seasonal, decomposition.resid

    def acf(self):

        acfVal = acf(self.tsValue)
        return acfVal

    def pacf(self):

        pacfVal = pacf(self.tsValue)
        return pacfVal


if __name__ == '__main__':

    filename = "data/1.csv"
    summary = DataSummary()
    summary.loadData(filename)

    # 统计特征
    statisticInfo = summary.getStatisticInfo()
    print(statisticInfo.getInfo())

    # 平稳性及随机性
    stationarity_p = summary.stationarity()
    print(stationarity_p)
    randomness_p = summary.randomness(lags=1)
    print(randomness_p)

    # 差分后平稳性及随机性
    summary.diff(order=1)
    stationarity_p = summary.stationarity()
    print(stationarity_p)
    randomness_p = summary.randomness(lags=1)
    print(randomness_p)

    # 分解
    #trend, seasonal, residual = summary.decomposition(freq=10)

    # acf和pacf
    acfVal = summary.acf()
    print(acfVal)
    pacfVal = summary.pacf()
    print(pacfVal)






