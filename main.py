# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-17T13:41:55.724387Z","iopub.execute_input":"2024-03-17T13:41:55.724720Z","iopub.status.idle":"2024-03-17T13:41:56.785607Z","shell.execute_reply.started":"2024-03-17T13:41:55.724685Z","shell.execute_reply":"2024-03-17T13:41:56.783074Z"}}
import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
import os
import errno

from sklearn.model_selection import train_test_split

from argparse import ArgumentParser, Namespace


class PointEstimator:
    def __init__(self):
        self.input_file_path = "kaggle/input/winemag-data-130k-v2.csv"
        self.dataframe = None

    def read_dataframe(self):
        self.dataframe = pd.read_csv(self.input_file_path)

    def preprocess_dataframe(self):
        new_columns = {"Unnamed: 0": "id"}
        self.dataframe.rename(columns=new_columns, inplace=True)


def main():
    estimator = PointEstimator()

    estimator.read_dataframe()

    estimator.preprocess_dataframe()

if __name__ == "__main__":
    main()
