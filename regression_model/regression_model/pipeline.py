from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from regression_model.processing import preprocessors as pp
from regression_model.config import config

marathon_pipeline = Pipeline(
    [
        (
            "categorical_imputer",
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA),
        ),
        (
            "rare_label_encoder",
            pp.RareLabelCategoricalEncoder(variables=config.CATEGORICAL_VARS),
        ),
        (
            "ordinal_encoder",
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)
        ),
        (
            "regressor",
            RandomForestRegressor(max_depth=4, n_estimators=100, random_sate=0)
        )
    ]
)
