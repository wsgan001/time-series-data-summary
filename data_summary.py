import pandas as pd
import numpy as np
from load_data import *
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels as sm
from scipy.stats import skew, kurtosis
import scipy.signal as signal
import matplotlib.pyplot as plt
import peakutils


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
        self.tile5 = None
        self.tile95 = None
        self.skewness = None
        self.kurtosis = None


    def getInfo(self):

        return {"num": self.num, "mean": self.mean, "std": self.std, "min": self.minNum,
                "max": self.maxNum, "median": self.median, "unique_num": self.uniqueNum, "null_num": self.nullNum,
                "95_tile": self.tile95, "5_tile": self.tile5, "skewness": self.skewness, "kurtosis": self.kurtosis}

class DataSummary():

    def __init__(self, ts):

        assert isinstance(ts, pd.DataFrame), "Invalid data format"
        self.ts = ts
        assert "value" in ts.columns, "time series has no value"
        self.tsValue = ts["value"]


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
        statisticInfo.tile5 = np.percentile(self.tsValue, 5)
        statisticInfo.tile95 = np.percentile(self.tsValue, 95)
        statisticInfo.kurtosis = kurtosis(self.tsValue)
        statisticInfo.skewness = skew(self.tsValue)

        return statisticInfo

    def getTrend(self, order, curoff, inplace=False):

        assert order > 0, "order must be positive"
        # N = 2  # Filter order
        # Wn = 0.01  # Cutoff frequency
        B, A = signal.butter(order, curoff, output='ba')

        trend = signal.filtfilt(B, A, self.tsValue)
        if inplace:
            self.tsValue = trend
        else:
            return trend

    def getPeakIndex(self, threshold):

        indices = peakutils.indexes(self.tsValue, thres=threshold, min_dist=0.1)
        return indices

    def stationarity(self):

        ADF_p_value = adfuller(self.tsValue, autolag='AIC')[1]
        return ADF_p_value

    def randomness(self, lags):

        assert lags > 0, "lags must be positive"
        p_value = acorr_ljungbox(self.tsValue, lags=lags)[1][0]
        return p_value

    def diff(self, order, inplace=False):

        assert order > 0, "order must be positive"
        tsDiff = np.diff(self.tsValue, n=order)
        if inplace:
            self.tsValue = tsDiff
        else:
            return tsDiff

    def decomposition(self, freq):

        decomposition = seasonal_decompose(self.tsValue, model="additive", )  # 季节分解
        return decomposition.trend, decomposition.seasonal, decomposition.resid

    def acf(self):

        acfVal, confint = acf(self.tsValue, alpha=0.05)
        return acfVal, confint

    def pacf(self):

        pacfVal, confint  = pacf(self.tsValue, alpha=0.05)
        return pacfVal, confint


if __name__ == '__main__':

    filename = "data/0.csv"
    ts = loadTimeSeries(filename)
    summary = DataSummary(ts)
    #plt.plot(summary.tsValue)
    #plt.show()

    # 统计特征
    statisticInfo = summary.getStatisticInfo()
    print(statisticInfo.getInfo())

    # butterworth滤波得到trend
    trend = summary.getTrend(order=6, curoff=0.5)
    plt.plot(trend)
    plt.show()

    # 寻找peak值位置
    peakIndex = summary.getPeakIndex(threshold=0.1)
    print(peakIndex)
    plt.plot(summary.tsValue)
    plt.scatter(peakIndex, summary.tsValue[peakIndex], c="r")
    plt.show()

    # acf和pacf
    acfVal, confint = summary.acf()
    print(acfVal)
    print(confint)
    plt.bar(range(len(acfVal)), acfVal)
    plt.show()
    pacfVal, confint = summary.pacf()
    print(pacfVal)
    print(confint)
    plt.bar(range(len(pacfVal)), pacfVal)
    plt.show()

    # 自带工具绘制acf和pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(summary.tsValue, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(summary.tsValue, lags=40, ax=ax2)
    plt.show()

    # 平稳性及随机性
    stationarity_p = summary.stationarity()
    print(stationarity_p)
    randomness_p = summary.randomness(lags=1)
    print(randomness_p)

    # 差分后平稳性及随机性
    summary.diff(order=1, inplace=True)
    plt.plot(summary.tsValue)
    plt.show()
    stationarity_p = summary.stationarity()
    print(stationarity_p)
    randomness_p = summary.randomness(lags=1)
    print(randomness_p)

    # acf和pacf
    acfVal, confint = summary.acf()
    print(acfVal)
    print(confint)
    plt.bar(range(len(acfVal)), acfVal)
    plt.show()
    pacfVal, confint = summary.pacf()
    print(pacfVal)
    print(confint)
    plt.bar(range(len(pacfVal)), pacfVal)
    plt.show()

    # 自带工具绘制acf和pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(summary.tsValue, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(summary.tsValue, lags=40, ax=ax2)
    plt.show()

    # 分解
    #trend, seasonal, residual = summary.decomposition(freq=10)








