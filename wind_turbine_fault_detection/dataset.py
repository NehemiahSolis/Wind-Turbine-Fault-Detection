from pathlib import Path
import os

import typer
from loguru import logger
from tqdm import tqdm

from wind_turbine_fault_detection.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import polars as pl
from functools import partial

app = typer.Typer()


def get_project_root() -> str:
    """
    Returns the project root directory based on the location of this script.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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

        self.project_root = get_project_root()

        if self.verbose:
            print(f"Project root: {self.project_root}")

        base_data_path = os.path.join(self.project_root, RAW_DATA_DIR)

        self.base_data_path = os.path.normpath(os.path.join(self.project_root, base_data_path))
        if self.verbose:
            print(f"Resolved base data path: {self.base_data_path}")

    def import_data(self) -> dict:
        """
        Returns the Guck et al. data as a dictionary of Polars DataFrames for the specified wind farm.

        Returns:
            wind_farm_dict (dict): The dictionary containing the wind farm datasets.
        """
        if self.verbose:
            print(f"Importing data for wind farm: {self.wind_farm}")
            
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
        if self.verbose:
            print(f"Starting import for wind farm {wind_farm}")

        base_path = f"{self.base_data_path}/anomaly_detection_data/Wind Farm {wind_farm}/"
        datasets_path = os.path.join(base_path, "datasets")

        if not os.path.exists(datasets_path):
            raise FileNotFoundError(f"The path {datasets_path} does not exist.")

        lazy_read = partial(pl.scan_csv, separator=";")

        wind_farm_dict = {}

        for file in os.listdir(datasets_path):
            if file.endswith(".csv"):
                key = f"dataset_{os.path.splitext(file)[0]}"
                wind_farm_dict[key] = lazy_read(os.path.join(datasets_path, file))
                # create new column for "event_id" = {os.path.splitext(file)[0]} in wind_farm_dict[key]
                # convert "event_id" to integer
                wind_farm_dict[key] = wind_farm_dict[key].with_columns(
                    pl.lit(os.path.splitext(file)[0]).cast(pl.Int64).alias("event_id")
                )
                

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
                    pl.col("time_stamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
                )

        return wind_farm_dict

    def get_data(self, collect: bool = True) -> dict:
        """
        Returns the Guck et al. data as a dictionary of polars DataFrames for the specified wind farm.

        Args:
            collect (bool, optional): Whether to collect lazy frames. Defaults to True.

        Returns:
            wind_farm_dict (dict): The dictionary containing the wind farm datasets.
        """
        # Import data
        wind_farm_dict = self.import_data()

        # If collect is True, collect lazy frames
        if collect:
            self._collect_lazy_frames(wind_farm_dict)

        return wind_farm_dict

    @staticmethod
    def _collect_lazy_frames(wind_farm_dict: dict) -> None:
        """
        Collects lazy frames in the wind_farm_dict dictionary.

        Args:
            wind_farm_dict (dict): The dictionary containing the wind farm datasets.
        """
        # Iterate over the keys in wind_farm_dict
        for key in wind_farm_dict:
            # If the value is a LazyFrame, collect it
            if isinstance(wind_farm_dict[key], pl.LazyFrame):
                wind_farm_dict[key] = wind_farm_dict[key].collect()
            # If the value is a dictionary, check for nested LazyFrames
            elif isinstance(wind_farm_dict[key], dict):
                for sub_key in wind_farm_dict[key]:
                    if isinstance(wind_farm_dict[key][sub_key], pl.LazyFrame):
                        wind_farm_dict[key][sub_key] = wind_farm_dict[key][sub_key].collect()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    wind_farm: str = "ALL",
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset for wind farm {wind_farm}...")
    guck = Guck_et_al(wind_farm)
    data = guck.import_data()
    # Process the data as needed
    for key, df in data.items():
        logger.info(f"Processing {key}...")
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
            print(df.head())
        elif isinstance(df, dict):
            for sub_key, sub_df in df.items():
                if isinstance(sub_df, pl.LazyFrame):
                    sub_df = sub_df.collect()
                    df[sub_key] = sub_df
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
