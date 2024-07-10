from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from wind_turbine_fault_detection.config import PROCESSED_DATA_DIR
import numpy as np
import polars as pl
import pandas as pd
from scipy import interpolate
from scipy.interpolate import BSpline

app = typer.Typer()


def BSplineBasis(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Return B-Spline basis. Python equivalent to bs in R or the spmak/spval combination in MATLAB.
    This function acts like the R command bs(x,knots=knots,degree=degree, intercept=False)
    Arguments:
        x: Points to evaluate spline on, sorted increasing
        knots: Spline knots, sorted increasing
        degree: Spline degree.
    Returns:
        B: Array of shape (x.shape[0], len(knots)+degree+1).
    Note that a spline has len(knots)+degree coefficients. However, because the intercept is missing
    you will need to remove the last 2 columns. It's being kept this way to retain compatibility with
    both the matlab spmak way and how R's bs works.

    If K = length(knots) (includes boundary knots)
    Mapping this to R's bs: (Props to Nate Bartlett )
    bs(x,knots,degree,intercept=T)[,2:K+degree] is same as BSplineBasis(x,knots,degree)[:,:-2]
    BF = bs(x,knots,degree,intercept=F) drops the first column so BF[,1:K+degree] == BSplineBasis(x,knots,degree)[:,:-2]
    """
    nKnots = knots.shape[0]
    lo = min(x[0], knots[0])
    hi = max(x[-1], knots[-1])
    augmented_knots = np.append(np.append([lo] * degree, knots), [hi] * degree)
    DOF = nKnots + degree + 1  # DOF = K+M, M = degree+1
    spline = BSpline(augmented_knots, np.eye(DOF), degree, extrapolate=False)
    B = spline(x)
    return B


def groups_array(group_type: str, X: pl.DataFrame) -> np.ndarray:
    """
    Get array of feature groups for a particular type.

    Args:
        group_type (str): Type of feature groups to get. Must be 'feature' or 'turbine'.
        X (pl.DataFrame): Input feature matrix as a Polars DataFrame.

    Returns:
        np.ndarray: Array of feature groups.
    """
    # Validate group_type
    assert group_type in ["feature", "turbine"], "group_type must be one of 'feature' or 'turbine'"

    # Initialize an empty list to store the group IDs
    groups = []

    # Iterate over the columns (features) of the input matrix
    for feature in X.columns:
        # Iterate over the feature groups dictionary
        for group_id, feature_list in enumerate(create_group_dict(group_type).values()):
            # Check if the current feature is in the feature group
            if feature in feature_list:
                # Add the group ID to the list and break the loop
                groups.append(group_id)
                break
    
    # Return the array of feature groups
    return np.array(groups)


def create_group_dict(group_type: str) -> dict:
    """
    Create a dictionary of feature groups.

    Args:
        group_type (str): The type of feature groups to create. Must be 'feature' or 'turbine'.
    Returns:
        dict: A dictionary of feature groups.
    """
    if group_type == "feature":
        # Feature groups dictionary
        feature_groups = {
            "Ambient Conditions": [  # Ambient temperature, wind direction, wind speed, nacelle and nose cone temperatures
                "sensor_0_avg",
                "sensor_1_avg",
                "sensor_2_avg",
                "wind_speed_3_avg",
                "wind_speed_3_max",
                "wind_speed_3_min",
                "wind_speed_3_std",
                "wind_speed_4_avg",
                "sensor_43_avg",
                "sensor_53_avg",
            ],
            "Controller Temperatures": [  # Temperatures in the hub controller, top nacelle controller, and VCS-section
                "sensor_6_avg",
                "sensor_7_avg",
                "sensor_8_avg",
                "sensor_9_avg",
                "sensor_10_avg",
            ],
            "Component Temperatures": [  # Temperatures in various components
                "sensor_11_avg",
                "sensor_12_avg",
                "sensor_13_avg",
                "sensor_14_avg",
                "sensor_15_avg",
                "sensor_16_avg",
                "sensor_17_avg",
                "sensor_18_avg",
                "sensor_18_max",
                "sensor_18_min",
                "sensor_18_std",
                "sensor_19_avg",
                "sensor_20_avg",
                "sensor_21_avg",
                "sensor_35_avg",
                "sensor_36_avg",
                "sensor_37_avg",
                "sensor_38_avg",
                "sensor_39_avg",
                "sensor_40_avg",
                "sensor_41_avg",
            ],
            "Electrical Measurements": [  # Electrical measurements
                "sensor_22_avg",
                "sensor_23_avg",
                "sensor_24_avg",
                "sensor_25_avg",
                "sensor_26_avg",
                "sensor_32_avg",
                "sensor_33_avg",
                "sensor_34_avg",
            ],
            "Power Measurements": [  # Power measurements
                "reactive_power_27_avg",
                "reactive_power_27_max",
                "reactive_power_27_min",
                "reactive_power_27_std",
                "reactive_power_28_avg",
                "reactive_power_28_max",
                "reactive_power_28_min",
                "reactive_power_28_std",
                "power_29_avg",
                "power_29_max",
                "power_29_min",
                "power_29_std",
                "power_30_avg",
                "power_30_max",
                "power_30_min",
                "power_30_std",
                "sensor_31_avg",
                "sensor_31_max",
                "sensor_31_min",
                "sensor_31_std",
                "sensor_44",
                "sensor_45",
                "sensor_46",
                "sensor_47",
                "sensor_48",
                "sensor_49",
                "sensor_50",
                "sensor_51",
            ],
            "Operational Metrics": [  # Operational metrics
                "sensor_5_avg",
                "sensor_5_max",
                "sensor_5_min",
                "sensor_5_std",
                "sensor_42_avg",
                "sensor_52_avg",
                "sensor_52_max",
                "sensor_52_min",
                "sensor_52_std",
            ],
        }
        return feature_groups
    else:
        raise ValueError("group_type must be one of 'feature' or 'turbine'")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
