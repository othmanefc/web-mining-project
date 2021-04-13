import pandas as pd


def load_data(path_data, split=True, is_csv=True):
    if is_csv:
        data = pd.read_csv(path_data)
    else:
        data = pd.read_feather(path_data)
    if split:
        return data[~data.ups.isnull()], data[data.ups.isnull()]
    return data