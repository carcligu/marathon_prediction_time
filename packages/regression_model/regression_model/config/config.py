import pathlib
import regression_model
import pandas as pd

PACKAGE_ROOT = pathlib.Path(regression_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"
DATA_FILE = "MarathonData.csv"
TARGET = 'MarathonTime'

CATEGORICAL_VARS_WITH_NA = ['Category', 'CrossTraining']

CATEGORICAL_VARS = ['Category', 'CrossTraining']

FEATURES = ['Category', 'km4week', 'sp4week', 'CrossTraining']

VARS_NA_NOT_ALLOWED = ['km4week', 'sp4week']

PIPELINE_NAME = "marathon_regression"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05