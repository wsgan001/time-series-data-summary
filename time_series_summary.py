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
from base_summary import *

class TimeSeriesSummary(Summary):

    def __init__(self, dataset):

        super().__init__(dataset)

    def getStatisticInfo(self):

        basicInfo = {}

        for feature in self.digitalFeatures:

            tsValue = self.dataset[feature]

            num = int(len(tsValue))
            mean = float(np.mean(tsValue))
            std = float(np.std(tsValue))
            minNum = float(np.min(tsValue))
            maxNum = float(np.max(tsValue))
            median = float(np.median(tsValue))
            uniqueNum = int(tsValue.nunique())
            nullNum = int(pd.isnull(tsValue).sum())
            tile5 = float(np.percentile(tsValue, 5))
            tile95 = float(np.percentile(tsValue, 95))
            kurto = float(kurtosis(tsValue))
            skewness = float(skew(tsValue))

            stationary_p = adfuller(tsValue, autolag='AIC')[1]
            random_p = acorr_ljungbox(tsValue, lags=1)[1][0]

            diff_tsValue = np.diff(tsValue, n=1)

            stationary_p_diff = adfuller(diff_tsValue, autolag='AIC')[1]
            random_p_diff = acorr_ljungbox(diff_tsValue, lags=1)[1][0]

            basicInfo[feature] = {"num": num, "mean": mean, "std": std, "min": minNum,
                    "max": maxNum, "median": median, "unique_num": uniqueNum, "null_num": nullNum,
                    "95_tile": tile95, "5_tile": tile5, "skewness": skewness,
                "kurtosis": kurto, "stationary": stationary_p, "randomness": random_p,
                        "stationary_diff":stationary_p_diff, "randomness_diff": random_p_diff }

        return basicInfo

    def getTrend(self, order, curoff):

        trendDict = {}

        for feature in self.digitalFeatures:

            # N = 2  # Filter order
            # Wn = 0.01  # Cutoff frequency
            B, A = signal.butter(order, curoff, output='ba')

            trend = signal.filtfilt(B, A, self.dataset[feature])

            trendDict[feature] = trend.tolist()

        return trendDict

    def getPeakIndex(self, threshold):

        peakIndex = {}

        for feature in self.digitalFeatures:

            indices = peakutils.indexes(self.dataset[feature], thres=threshold)
            peakIndex[feature] = indices.tolist()

        return peakIndex


    def decomposition(self, freq):

        decomposition = seasonal_decompose(self.tsValue, model="additive", )  # 季节分解
        return decomposition.trend, decomposition.seasonal, decomposition.resid

    def getAcf(self):

        acfDict = {}

        for feature in self.digitalFeatures:

            acfVal, confint = acf(self.dataset[feature], alpha=0.05)
            acfDict[feature] = [acfVal.tolist(), confint.tolist()]

        return acfDict

    def getPacf(self):

        pacfDict = {}

        for feature in self.digitalFeatures:

            pacfVal, confint  = pacf(self.dataset[feature], alpha=0.05)
            pacfDict[feature] = [pacfVal.tolist(), confint.tolist()]

        return pacfDict

    def getCorrTimeSeries(self):

        corr = self.dataset[self.digitalFeatures].corr()  # 直接计算不同连续值特征pairwise, 得到相关性矩阵
        return {"corr": corr.values.tolist()}


if __name__ == '__main__':

    dataset = loadData('data/deepAD_data_summary_test_data1.csv')
    summary = TimeSeriesSummary(dataset)

    corr = summary.getCorrTimeSeries()
    print(corr)

    # basicInfo = summary.getStatisticInfo()
    # trendDict = summary.getTrend(order=5, curoff=0.5)
    # peakIndex = summary.getPeakIndex(threshold=0.1)
    # acfDict = summary.getAcf()
    # pacfDict = summary.getPacf()
    #
    # for feature in summary.digitalFeatures:
    #
    #     print("feature:", feature)
    #
    #     plt.plot(dataset[feature])
    #     plt.show()
    #
    #     # 统计特征,平稳性、随机性
    #
    #     print(basicInfo[feature])
    #
    #     # butterworth滤波得到trend
    #     plt.plot(trendDict[feature])
    #     plt.show()
    #
    #     # 寻找peak值位置
    #     plt.plot(dataset[feature])
    #     plt.scatter(peakIndex[feature], dataset[feature][peakIndex[feature]], c="r")
    #     plt.show()
    #
    #     # acf和pacf
    #     print(acfDict[feature])
    #     plt.bar(range(len(acfDict[feature][0])), np.array(acfDict[feature][0]))
    #     plt.show()
    #     print(pacfDict[feature])
    #     plt.bar(range(len(pacfDict[feature][0])), np.array(pacfDict[feature][0]))
    #     plt.show()
    #
    #     # 自带工具绘制acf和pacf
    #     from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    #     fig = plt.figure(figsize=(12, 8))
    #     ax1 = fig.add_subplot(211)
    #     fig = plot_acf(dataset[feature], lags=40, ax=ax1)
    #     ax2 = fig.add_subplot(212)
    #     fig = plot_pacf(dataset[feature], lags=40, ax=ax2)
    #     plt.show()










