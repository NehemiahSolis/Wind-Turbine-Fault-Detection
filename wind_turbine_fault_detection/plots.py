from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt

from wind_turbine_fault_detection.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def create_labels(classes, samples_per_class):
    """
    Generates labels for a given number of classes and samples per class.

    Args:
        classes (int): The number of classes.
        samples_per_class (int): The number of samples per class.

    Returns:
        numpy.ndarray: An array of labels.
    """
    # Repeat the range of classes to match the number of samples per class
    # and return the resulting array.
    return np.repeat(np.arange(classes), samples_per_class)



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
