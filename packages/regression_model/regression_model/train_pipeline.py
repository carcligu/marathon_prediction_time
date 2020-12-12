from sklearn.model_selection import train_test_split

from regression_model import pipeline
from regression_model.processing.data_management import load_dataset, save_pipeline
from regression_model.config import config
from regression_model.predict import make_prediction


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=0
    )  # we are setting the seed here

    pipeline.marathon_pipeline.fit(X_train[config.FEATURES], y_train)

    save_pipeline(pipeline_to_persist=pipeline.marathon_pipeline)

    print("########################################")
    print("Test prediction: ")

    test_data = load_dataset(file_name='test.csv')
    single_test_json = test_data[0:1].to_json(orient='records')

    # When
    subject = make_prediction(input_data=single_test_json)
    print(subject)

if __name__ == "__main__":
    print("Training pipeline...")
    run_training()