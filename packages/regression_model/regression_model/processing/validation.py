from regression_model.config import config
import pandas as pd

def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:

    validated_data = input_data.copy()

    if input_data[config.VARS_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.VARS_NA_NOT_ALLOWED
        )

    return validated_data