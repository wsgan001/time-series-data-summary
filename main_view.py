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
parser.add_argument('--filepath', '-fp', type=str, help='resume from checkpoint')
args = parser.parse_args()

def main():

    filepath = args.filepath

    try:
        dataset = loadData(filepath)
    except:
        return {"error": "file open failed"}

    if args.run == "table_basic_summary":

        summary = TableSummary(dataset)
        basicInfo = summary.getBasicInformation()
        baiscInfo_json = json.dumps(basicInfo)
        print(baiscInfo_json)

    if args.run == "column_recommend":

        summary = TableSummary(dataset)
        recommend = summary.getColumnRecommend()
        recommend_json = json.dumps(recommend)
        print(recommend_json)

    if args.run == "column_basic_summary":

        summary = ColumnsSummary(dataset)
        basicInfo = summary.getBasicInformation()
        baiscInfo_json = json.dumps(basicInfo)
        print(baiscInfo_json)

    # if args.run == "column_hist":
    #
    #     summary = ColumnsSummary(dataset)
    #     hist_json = json.dumps(summary.hist)
    #     print(hist_json)

    if args.run == "time_series_basic_summary":

        summary = TimeSeriesSummary(dataset)
        basicInfo = summary.getStatisticInfo()
        baiscInfo_json = json.dumps(basicInfo)
        print(baiscInfo_json)

    if args.run == "time_series_trend":
        summary = TimeSeriesSummary(dataset)
        trend = summary.getTrend(order=5, curoff=0.5)
        trend_json = json.dumps(trend)
        print(trend_json)

    if args.run == "time_series_peak":

        summary = TimeSeriesSummary(dataset)
        peak = summary.getPeakIndex(threshold=0.1)
        peak_json = json.dumps(peak)
        print(peak_json)

    if args.run == "time_series_acf":

        summary = TimeSeriesSummary(dataset)
        acfDict = summary.getAcf()
        acfDict_json = json.dumps(acfDict)
        print(acfDict_json)

    if args.run == "time_series_pacf":

        summary = TimeSeriesSummary(dataset)
        pacfDict = summary.getPacf()
        pacfDict_json = json.dumps(pacfDict)
        print(pacfDict_json)







if __name__ == '__main__':
    main()




