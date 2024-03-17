# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-17T13:41:55.724387Z","iopub.execute_input":"2024-03-17T13:41:55.724720Z","iopub.status.idle":"2024-03-17T13:41:56.785607Z","shell.execute_reply.started":"2024-03-17T13:41:55.724685Z","shell.execute_reply":"2024-03-17T13:41:56.783074Z"}}
import numpy as np
import pandas as pd
import os
import errno

from argparse import ArgumentParser, Namespace


class PointEstimator:
    def __init__(self):
        self.input_file_path = "kaggle/input/winemag-data-130k-v2.csv"
        self.dataframe = None

    def read_dataframe(self):
        self.dataframe = pd.read_csv(self.input_file_path)


def main():
    estimator = PointEstimator()

    estimator.read_dataframe()


if __name__ == "__main__":
    main()
