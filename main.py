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
        self.INPUT_FILE_PATH = "kaggle/input/winemag-data-130k-v2.csv"
        self.TEST_SIZE = 0.2
        self.X_COLUMNS = ["description"]
        self.Y_COLUMNS = ["points"]
        self.MAX_TRIALS = 1
        self.N_EPOCHS = 2

        self.dataframe = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def read_dataframe(self):
        self.dataframe = pd.read_csv(self.INPUT_FILE_PATH)

    def preprocess_dataframe(self):
        new_columns = {"Unnamed: 0": "id"}
        self.dataframe.rename(columns=new_columns, inplace=True)

    def prepare_data(self):
        train, test = train_test_split(self.dataframe, test_size=self.TEST_SIZE)

        self.x_train = train[self.X_COLUMNS].to_numpy()
        self.y_train = train[self.Y_COLUMNS].to_numpy()

        self.x_test = test[self.X_COLUMNS].to_numpy()
        self.y_test = test[self.Y_COLUMNS].to_numpy()

    def run_model(self):
        regressor = ak.TextRegressor(overwrite=True, max_trials=self.MAX_TRIALS)

        regressor.fit(self.x_train, self.y_train, epochs=self.N_EPOCHS)
        y_predicted = regressor.predict(self.x_test)

        print(regressor.evaluate(self.x_test, self.y_test))


def main():
    estimator = PointEstimator()

    estimator.read_dataframe()
    estimator.preprocess_dataframe()
    estimator.prepare_data()
    estimator.run_model()


if __name__ == "__main__":
    main()
