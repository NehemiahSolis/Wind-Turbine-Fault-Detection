"""
Allows you to import the data into the notebook or python script.
If you don't have the data, download it from the link produced by the get_data.py script
"""

import os
import pandas as pd
import polars as pl  # type: ignore
from functools import partial


class Guck_et_al:
    """
    Class for importing and processing Guck et al. data.

    Comes from the massive dataset compiled by Christian Guck, Cyriana Roelofs, and Stefan Faulstich. The data consists of 95 datasets,
    containing 89 years of SCADA time series distributed across 36 different wind turbines from the three wind farms A, B and C.
    The number of features depends on the wind farm; Wind farm A has 86 features, wind farm B has 257 features and wind farm C has 957 features.

    The overall dataset is balanced, as 44 out the 95 datasets contain a labeled anomaly event that leads up to a turbine fault and the other 51
    datasets represent normal behavior. Additionally, the quality of training data is ensured by turbine-status-based labels for each data point and
    further information about some of the given turbine faults are included.

    The data for Wind farm A is based on data from the EDP open data platform (https://www.edp.com/en/innovation/open-data/data),
    and consists of 5 wind turbines of an onshore wind platform in Portugal. It contains SCADA data and information derived by a given fault logbook
    which defines start timestamps for specified faults. From this data 22 datasets were selected to be included in this data collection.
    The other two wind farms are offshore wind farms located in Germany. All three datasets were anonymized due to confidentiality reasons for the wind farms B and C.
    Each dataset is provided in form of a csv-file with columns defining the features and rows representing the data points of the time series.

    Attributes:
        wind_farm (str): The wind farm name, either 'A', 'B', or 'C'.
        verbose (bool): Whether to print verbose output. Default is False.
        wind_farm_dict (dict): The dictionary containing the wind farm datasets.
    """

    def __init__(self, wind_farm: str, verbose: bool = False) -> None:
        """
        Initializes the Guck et al. data and enforces the correct format of the 'duration' column.

        Args:
            wind_farm (str): The wind farm name, either 'A', 'B', or 'C'. Default is 'All', which includes all wind farms in a dictionary of dictionaries.
            verbose (bool, optional): Whether to print verbose output. Default is False.
        """
        self.verbose = verbose
        self.wind_farm = wind_farm.upper()
        if self.wind_farm not in ["A", "B", "C", "ALL"]:
            raise ValueError("Wind farm must be either 'A', 'B', 'C', or 'All'.")

    def import_data(self) -> dict:
        """
        Returns the Guck et al. data as a dictionary of Polars DataFrames for the specified wind farm.

        Returns:
            wind_farm_dict (dict): The dictionary containing the wind farm datasets.
        """
        if self.wind_farm == "ALL":
            return self._import_all_wind_farms()

        return self._import_single_wind_farm(self.wind_farm)

    def _import_all_wind_farms(self) -> dict:
        """
        Imports data for all wind farms (A, B, and C).

        Returns:
            wind_farm_dict (dict): A dictionary where keys are wind farm identifiers and values are dictionaries
                                   containing the data for each wind farm.
        """
        return {farm: self._import_single_wind_farm(farm) for farm in ["A", "B", "C"]}

    def _import_single_wind_farm(self, wind_farm: str) -> dict:
        """
        Imports data for a single specified wind farm.

        Args:
            wind_farm (str): The identifier of the wind farm to import data for.

        Returns:
            wind_farm_dict (dict): A dictionary containing the datasets for the specified wind farm.
        """
        base_path = f"data/anomaly_detection_data/Wind Farm {wind_farm}"
        datasets_path = os.path.join(base_path, "datasets")

        lazy_read = partial(pl.scan_csv, separator=";")

        wind_farm_dict = {}

        for file in os.listdir(datasets_path):
            if file.endswith(".csv"):
                key = f"dataset_{os.path.splitext(file)[0]}"
                wind_farm_dict[key] = lazy_read(os.path.join(datasets_path, file))

        # import event_info and feature_description
        for file in ["event_info.csv", "feature_description.csv"]:
            key = f"{os.path.splitext(file)[0]}"
            wind_farm_dict[key] = lazy_read(os.path.join(base_path, file))

        # clean and process data
        wind_farm_dict["feature_description"] = self._clean_feature_descriptions(
            wind_farm_dict["feature_description"]
        )
        wind_farm_dict = self.convert_to_datetime(wind_farm_dict)

        return wind_farm_dict

    @staticmethod
    def _clean_feature_descriptions(feature_descriptions: pl.LazyFrame) -> pl.LazyFrame:
        """
        Cleans the feature descriptions DataFrame by replacing specific characters.

        Args:
            feature_descriptions (pl.LazyFrame): The DataFrame containing the feature descriptions to be cleaned.
        """
        return feature_descriptions.with_columns(
            pl.col("unit").str.replace("�C", "°C").str.replace("�", "°")
        )

    @staticmethod
    def convert_to_datetime(wind_farm_dict: dict) -> dict:
        """
        Converts the 'event_start' column in 'event_info' dataset and 'time_stamp' columns in all 'datasets_{}' to datetime objects.

        Args:
            wind_farm_dict (dict): The dictionary containing the wind farm datasets.

        Returns:
            wind_farm_dict (dict): The dictionary with the datetime conversions applied.
        """
        datetime_cols = ["event_start", "event_end"]

        wind_farm_dict["event_info"] = wind_farm_dict["event_info"].with_columns(
            [
                pl.col(col).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
                for col in datetime_cols
            ]
        )

        for key in wind_farm_dict:
            if key.startswith("dataset_"):
                wind_farm_dict[key] = wind_farm_dict[key].with_columns(
                    pl.col("time_stamp").str.strptime(
                        pl.Datetime, format="%Y-%m-%d %H:%M:%S"
                    )
                )

        return wind_farm_dict

    def get_data(self, collect: bool = True) -> dict:
        """
        Returns the Guck et al. data as a dictionary of pandas DataFrames for the specified wind farm.

        Returns:
            wind_farm_dict (dict): The dictionary containing the wind farm datasets.
        """
        wind_farm_dict = self.import_data()
        
        if collect:
            # Collect lazy frames
            for key in wind_farm_dict:
                if isinstance(wind_farm_dict[key], pl.LazyFrame):
                    wind_farm_dict[key] = wind_farm_dict[key].collect()
                elif isinstance(wind_farm_dict[key], dict):  # Check for nested dictionaries (e.g., when wind_farm == "ALL")
                    for sub_key in wind_farm_dict[key]:
                        if isinstance(wind_farm_dict[key][sub_key], pl.LazyFrame):
                            wind_farm_dict[key][sub_key] = wind_farm_dict[key][sub_key].collect()
                        
        return wind_farm_dict

class SCADA:
    """
    Class for importing and processing SCADA data.

    Attributes:
        scada (pd.DataFrame): The SCADA data as a pandas DataFrame.
        fault (pd.DataFrame): The fault data as a pandas DataFrame.
        status (pd.DataFrame): The status data as a pandas DataFrame.
        verbose (bool): Whether to print verbose output. Default is False.
    """

    def __init__(self, verbose=False):
        """
        Initializes the SCADA data and enforces the correct format of the 'DateTime' column.

        Args:
            verbose (bool, optional): Whether to print verbose output. Default is False.
        """
        self.verbose = verbose

        self.scada, self.fault, self.status = self.import_scada()
        if self.verbose:
            print("Columns in status (before enforcing): ", self.status.columns)
        self.enforce_datetimes()
        if self.verbose:
            print("Columns in status (after enforcing): ", self.status.columns)

    @staticmethod
    def import_scada():
        """
        Returns the SCADA data as three pandas DataFrames.

        Returns:
            scada (pd.DataFrame): The SCADA data as a pandas DataFrame.
            fault (pd.DataFrame): The fault data as a pandas DataFrame.
            status (pd.DataFrame): The status data as a pandas DataFrame.
        """
        fault = pd.read_csv("data/scada_data/fault_data.csv", sep=",")
        scada = pd.read_csv("data/scada_data/scada_data.csv", sep=",")
        status = pd.read_csv("data/scada_data/status_data.csv", sep=",")
        return scada, fault, status

    def enforce_datetimes(self):
        """
        Converts the 'DateTime' column of the SCADA data to the correct format if it is not already in the correct format.

        Returns:
            self: The updated SCADA object.
        """
        self.scada["DateTime"] = pd.to_datetime(self.scada["DateTime"], format="mixed")
        self.fault["DateTime"] = pd.to_datetime(self.fault["DateTime"], format="mixed")
        # Rename "Time" column to "DateTime"
        self.status.rename(columns={"Time": "DateTime"}, inplace=True)
        self.status["DateTime"] = pd.to_datetime(
            self.status["DateTime"], format="mixed"
        )

        return self

    def get_data(self):
        """
        Returns the SCADA data as three pandas DataFrames.

        Returns:
            scada (pd.DataFrame): The SCADA data as a pandas DataFrame.
            fault (pd.DataFrame): The fault data as a pandas DataFrame.
            status (pd.DataFrame): The status data as a pandas DataFrame.
        """
        if self.verbose:
            print("Data imported successfully.")
        return self.scada, self.fault, self.status


class WTPHM:
    """Class for importing and processing WTPHM data.

    Attributes:
        scada (pd.DataFrame): The WTPHM data as a pandas DataFrame.
        events (pd.DataFrame): The WTPHM events as a pandas DataFrame.

    """

    def __init__(self):
        """Initializes the WTPHM data and enforces the correct format of the 'duration' column."""
        self.scada, self.events = self.import_wtphm()
        self.create_duration()

    @staticmethod
    def import_wtphm():
        """Imports the WTPHM data as pandas DataFrames.

        Returns:
            scada (pd.DataFrame): The WTPHM scada data as a pandas DataFrame.
            events (pd.DataFrame): The WTPHM events data as a pandas DataFrame.
        """
        scada = pd.read_csv("data/wtphm_test_data/scada_data.csv", parse_dates=["time"])
        events = pd.read_csv(
            "data/wtphm_test_data/event_data.csv", parse_dates=["time_on", "time_off"]
        )
        return scada, events

    def create_duration(self):
        """Converts the 'duration' column of the WTPHM events data to the correct format if it is not already in the correct format.

        Returns:
            self: The updated WTPHM object.
        """
        self.events.duration = pd.to_timedelta(self.events.duration)
        return self

    def get_data(self):
        """Imports and creates the WTPHM data as pandas DataFrames.

        Returns:
            scada (pd.DataFrame): The WTPHM scada data as a pandas DataFrame.
            events (pd.DataFrame): The WTPHM events data as a pandas DataFrame.
        """
        return self.scada, self.events


def get_vibration():
    """Returns the vibration data as a pandas dataframe."""
    path = "data/vibration_analysis_data"
    data = {}
    for file in os.listdir(path):
        if file.endswith(".csv"):
            data[file] = pd.read_csv(os.path.join(path, file), sep=",")
        elif file.endswith(".xlsx"):
            data[file] = pd.read_excel(os.path.join(path, file))
    return data
