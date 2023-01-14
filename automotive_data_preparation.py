import os
import random
import pandas as pd

from auxiliaries import load_obj, save_obj, np

# Auxiliary function to convert curve values into timeseries data
def ts_conversion(data: pd.DataFrame, strip: str = "[]") -> pd.DataFrame:
    """Aux function that performs a conversion of inserted curve value strings to list like time series.
    - data : pd.DataFrame
        Pandas dataframe to perform the conversion.
    - strip : str
        Bracket type for handling formatted strings."""
    data.motor_id = pd.to_numeric(data["motor_id"])  # solves id inconsistency
    # Split string observations into list
    data.curve_values_string = [
        curve.replace("[", "{").replace("]", "}") for curve in data.curve_values_string
    ]
    data.curve_values_string = [
        curve.strip(strip).split(",") for curve in data.curve_values_string
    ]
    for i in range(len(data)):
        data.curve_values_string.iloc[i] = [
            float(val) for val in data.curve_values_string.iloc[i]
        ]
    return data


# Auxiliary function perform a number of preprocessing steps
def import_automotive_data_csv(file: str, afo: str) -> None:
    """Aux function that performs the following tasks for data preparation:
    - remove all not needed observations (by type and afo)
    - prepare the curve value strings to actual lists of data
    - normalize the time series to values between 0 and 1"""

    # load all known Anomalies as observations in a dataframe
    df = pd.read_csv("data/{f}.csv".format(f=file), delimiter=",", index_col=0)
    # limit observations to one AFO
    df = df[df["afo_nr"] == afo]
    # limit selection to show only curve values and reset index (back up)
    df = df[
        pd.Series(
            [
                "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0" not in string
                for string in df.curve_values_string
            ],
            name="bool",
        ).values
    ]

    # remove duplicates in the Motor IDs
    df = df.drop_duplicates("motor_id", keep="last").reset_index(drop=True)
    # perform data conversion to get time series data
    df = ts_conversion(df, "{}")
    # transform data to dictionary and remove unnecessary attributes
    df = df[["motor_id", "curve_values_string"]].set_index("motor_id").T.to_dict("list")

    # find the maximal value for the following normalization
    max_value = 0
    for key in df.keys():
        df[key] = df[key][0]  # unpack
        for value in df[key]:
            if value > max_value:
                max_value = value

    # normalize values to a value range of 0 to 1
    for key in df.keys():
        df[key] = [value / max_value for value in df[key]]

    # save the prepared dictionary using pickle
    save_obj(df, file, save_to="data")


def prepare_automotive_data(
    number_of_folds: int, ratio: float, target_path: str
) -> None:
    """Function to pepare and save the automotive data in train and test np.ndarrays.
    - number_of_folds : int:
        Split the OK samples in n unique sections.
    - ratio: float
        Ratio to split the train and test data. 
    - target_path : str
        Folder to store the prepared data in."""

    # load data from previously prepared dictionaries
    dict_nOK = load_obj("anomalies_l400")
    dict_OK = load_obj("observations_l400")

    # remove all anomalies from OK observations (just double checking)
    for key in dict_nOK.keys():
        dict_OK.pop(key, None)

    # limit OK observations to a total of 50.000 (run time restriction)
    dict_OK = {k: dict_OK[k] for k in list(dict_OK)[:50000]}

    # assign nOK-Label of 1 to all observations in the nOK dictionary
    dict_nOK = {key: [1, dict_nOK[key]] for key in dict_nOK.keys()}

    # split OK keys in ten different lists of same length (each 5.000 samples)
    OK_keys_listed = list(dict_OK.keys())
    maximal_length = len(OK_keys_listed)
    random.seed(42)
    OK_keys_shuffled = random.sample(OK_keys_listed, maximal_length)
    keys_by_interval = {}  # target dict
    # tter over n_folds to generate multiple scenarios (same nOK's with different OK's)
    for i in range(number_of_folds):
        a = int(maximal_length / number_of_folds) * i
        b = int(maximal_length / number_of_folds) * (i + 1)
        keys_by_interval["interval_" + str(i)] = OK_keys_shuffled[a:b]

    # prepare data accoding to the selected number of scenarios
    for interval in range(number_of_folds):
        # assign OK-Label of 0 to all observations in each of the OK dictionaries
        data_OK = {
            key: [0, dict_OK[key]]
            for key in keys_by_interval["interval_" + str(interval)]
        }

        # join OK samples with nOK samples
        data = {**data_OK, **dict_nOK}
        # assign x and y values accoding to dict
        y = [list(data.values())[i][0] for i in range(len(data.keys()))]  # binary label
        x = [list(data.values())[i][1] for i in range(len(data.keys()))]

        # zip, shuffle and un-zip
        _x_y = list(zip(x, y))
        random.shuffle(_x_y)
        x, y = zip(*_x_y)

        # perform train und test split
        interval_length = len(data)
        X_train = np.array(x[: int(ratio * interval_length)], dtype=object)
        y_train = np.array(y[: int(ratio * interval_length)], dtype=object)
        X_test = np.array(x[int(ratio * interval_length) :], dtype=object)
        y_test = np.array(y[int(ratio * interval_length) :], dtype=object)

        # reshape X values to match sklearn format (requires 3D-array)
        X_train = X_train.reshape(X_train.shape + (1,))
        X_test = X_test.reshape(X_test.shape + (1,))

        # update console
        print(
            f"{interval}: OK - TRAIN: {y_train.tolist().count(0)} | TEST: {y_test.tolist().count(0)}"
        )
        print(
            f"{interval}: nOK - TRAIN: {y_train.tolist().count(1)} | TEST: {y_test.tolist().count(1)}"
        )
        """
        # save prepared data using picke
        save_obj(
            [X_train, X_test, y_train, y_test],
            "train_test_" + str(interval),
            save_to=target_path,
        )"""


def remove_by_max_len(d: dict, max_len: int) -> dict:
    """Aux function to restrict a dict's observations according to a max length."""
    # Remove observations that are too long
    return {key: val for key, val in d.items() if len(val) < max_len}


if __name__ == "__main__":
    # import data from csv as pickle {'anomalies': nOK, 'observations': OK}
    files = ["anomalies", "observations"]
    for file in files:
        if not os.path.exists(f"data/{file}.pkl"):
            import_automotive_data_csv(file, "afo1")

    # remove observations according to the selected max length
    max_len = 400  # selected manually using a histogram
    if not os.path.exists("data/anomalies_l400.pkl"):
        save_obj(remove_by_max_len(load_obj("anomalies"), max_len), "anomalies_l400")
    if not os.path.exists("data/observations_l400.pkl"):
        save_obj(
            remove_by_max_len(load_obj("observations"), max_len), "observations_l400"
        )
    # perform n-fold split of the OK data and split train and test according to
    prepare_automotive_data(
        number_of_folds=10, ratio=0.7, target_path="data_prepared_automotive"
    )
