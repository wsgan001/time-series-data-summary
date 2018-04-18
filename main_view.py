import argparse
import pandas as pd
import numpy as np
from load_data import *
from table_summary import *
from column_summary import *
from time_series_summary import *
import json
import config

parser = argparse.ArgumentParser(description='DeepAD data summary')
parser.add_argument('--run', '-r', type=str, help='function to execute')
parser.add_argument('--filepath', '-fp', type=str, help='dataset csv file path')
parser.add_argument('--savepath', '-sp', default="/", type=str, help='save clean dataset csv file path')
parser.add_argument('--communication', '-com', default="cmd", type=str, help='communication method, cmd or file')
parser.add_argument('--filter', default=[], nargs='+', type=str, help='choose features you need')
parser.add_argument('--ignore', default=[], nargs='+', type=str, help='choose features you ignore')
#parser.add_argument('--time_series_scope', '-ts_scope', default=[], nargs='+', type=str, help='choose scope you need in time series analysis')

args = parser.parse_args()

def main():

    returnInfo = {} # 返回结果

    # 读取数据
    filepath = args.filepath
    try:
        dataset = loadData(filepath)
    except:
        print({"Error": "load data failed"})

    if args.run == "clean":  # 清洗数据
        savePath = args.savepath
        summary = Summary(dataset)
        summary.cleanData(savePath)

    # 特征过滤
    filterFeatures = args.filter
    if len(filterFeatures) != 0:
        dataset = dataset[filterFeatures]
    exceptFeatures = args.ignore
    if len(exceptFeatures) != 0:
        dataset = dataset.drop(exceptFeatures, axis=1)

    # 执行操作
    if args.run == "table_basic_summary":

        summary = TableSummary(dataset)
        returnInfo = summary.getBasicInformation()

    if args.run == "column_recommend":

        summary = TableSummary(dataset)
        returnInfo= summary.getColumnRecommend()

    if args.run == "column_basic_summary":

        summary = ColumnsSummary(dataset)
        returnInfo = summary.getBasicInformation()

    if args.run == "column_hist":

        summary = ColumnsSummary(dataset)
        returnInfo = summary.getHist()

    if args.run == "get_rank":

        summary = ColumnsSummary(dataset)
        returnInfo = summary.getRank()

    if args.run == "get_time_series":

        summary = TimeSeriesSummary(dataset)
        returnInfo = summary.getTimeSeries()

    if args.run == "time_series_basic_summary":

        summary = TimeSeriesSummary(dataset)
        returnInfo = summary.getStatisticInfo()

    if args.run == "time_series_trend":
        summary = TimeSeriesSummary(dataset)
        returnInfo = summary.getTrend(order=5, curoff=0.5)

    if args.run == "time_series_peak":

        summary = TimeSeriesSummary(dataset)
        returnInfo = summary.getPeakIndex(threshold=0.1)

    if args.run == "time_series_acf":

        summary = TimeSeriesSummary(dataset)
        returnInfo = summary.getAcf()

    if args.run == "time_series_pacf":

        summary = TimeSeriesSummary(dataset)
        returnInfo = summary.getPacf()

    if args.run == "cross_correlation":

        summary = TimeSeriesSummary(dataset)
        returnInfo = summary.getCorrTimeSeries()

    if args.run == "get_releationship":

        summary = ColumnsSummary(dataset)
        returnInfo = summary.getRealationShip()


    if args.run == "all":

        summary = TableSummary(dataset)
        returnInfo["table_summary"] = summary.getBasicInformation()
        returnInfo["recommend_columns"] = summary.getColumnRecommend()

        summary = ColumnsSummary(dataset)
        returnInfo["column_summary"] = summary.getBasicInformation()
        returnInfo["hist"] = summary.getHist()
        returnInfo["rank"] = summary.getRank()

        summary = TimeSeriesSummary(dataset)
        returnInfo["time_series_summary"] = summary.getStatisticInfo()
        returnInfo["time_series"] = summary.getTimeSeries()
        returnInfo["trend"] = summary.getTrend(order=5, curoff=0.5)
        returnInfo["peak_index"] = summary.getPeakIndex(threshold=0.1)
        returnInfo["acf"] = summary.getAcf()
        returnInfo["pacf"] = summary.getPacf()
        returnInfo["corr"] = summary.getCorrTimeSeries()

    if args.communication == "cmd":
        returnJsonStr = json.dumps(returnInfo)
        print(returnJsonStr)
    else:
        jsonFilePath = args.communication
        with open(jsonFilePath, "w") as outfile:
            json.dump(returnInfo, outfile)
        # with open("data/output.json", "r") as outfile:
        #     print(json.load(outfile))

if __name__ == '__main__':

    main()




