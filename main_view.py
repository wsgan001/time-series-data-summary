import argparse
import pandas as pd
import numpy as np
from load_data import *
from table_summary import *
from column_summary import *
from time_series_summary import *
import json

parser = argparse.ArgumentParser(description='DeepAD data summary')
parser.add_argument('--run', '-r', type=str, help='function to execute')
parser.add_argument('--filepath', '-fp', type=str, help='dataset csv file path')
parser.add_argument('--communication', '-com', default="cmd", type=str, help='communication method, cmd or file')
parser.add_argument('--filter', default=[], nargs='+', type=str, help='choose features you need')
#parser.add_argument('--time_series_scope', '-ts_scope', default=[], nargs='+', type=str, help='choose scope you need in time series analysis')

args = parser.parse_args()

def main():

    filepath = args.filepath

    try:
        filterFeatures = args.filter
        if len(filterFeatures) == 0:
            filterFeatures = None
        dataset = loadData(filepath, filterFeatures)
    except:
        return {"Error": "load data failed"}

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

    if args.communication == "file":
        with open("data/output.json", "w") as outfile:
            json.dump(returnInfo, outfile)
        # with open("data/output.json", "r") as outfile:
        #     print(json.load(outfile))
    elif args.communication == "cmd":
        returnJsonStr = json.dumps(returnInfo)
        print(returnJsonStr)


if __name__ == '__main__':
    main()




