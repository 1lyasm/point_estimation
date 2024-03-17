# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-03-17T13:41:55.724387Z","iopub.execute_input":"2024-03-17T13:41:55.724720Z","iopub.status.idle":"2024-03-17T13:41:56.785607Z","shell.execute_reply.started":"2024-03-17T13:41:55.724685Z","shell.execute_reply":"2024-03-17T13:41:56.783074Z"}}
import numpy as np
import pandas as pd
import os
import errno


class PointEstimator:
    def __init__(self):
        self.input_directory_path = "kaggle/input"
        self.input_file_paths = []
        self.dataframes = {}

    def fill_input_file_paths(self):
        if os.path.isdir(self.input_directory_path) is False:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self.input_directory_path
            )

        for directory_name, _, file_names in os.walk(self.input_directory_path):
            for file_name in file_names:
                file_path = os.path.join(directory_name, file_name)
                self.input_file_paths.append(file_path)

    def read_dataframes(self):
        for input_file_path in self.input_file_paths:
            _, extension = os.path.splitext(input_file_path)

            if extension == ".csv":
                self.dataframes[input_file_path] = pd.read_csv(input_file_path)
            elif extension == ".json":
                self.dataframes[input_file_path] = pd.read_json(input_file_path)


def main():
    estimator = PointEstimator()

    estimator.fill_input_file_paths()

    estimator.read_dataframes()
    print(f"Dataframes: {estimator.dataframes}")


if __name__ == "__main__":
    main()
