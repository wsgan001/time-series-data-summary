import argparse
import pandas as pd
import numpy as np
from load_data import *
from table_summary import *


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

    if args.run == "table_summary":

        summary = TableSummary(dataset)
        print(summary.baiscInfo)
        print(summary.recommendResult)


if __name__ == '__main__':
    main()




