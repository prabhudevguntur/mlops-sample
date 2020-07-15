import numpy as np
from code.training.train import train_model, feature_eng


def test_train_model():
    # Arrange
    data = pd.read_csv('sample.csv')
    y_train = data["median_house_value"].copy()
    X_train = data.drop("median_house_value", axis=1)

    # Act
    full_pipeline = feature_eng(X_train)
    housing_prepared = full_pipeline.fit_transform(X_train)
    reg_model = train_model(housing_prepared, y_train)

    # Assert
    preds = reg_model.predict(X_train)
    np.testing.assert_almost_equal(preds, [452600., 358500., 352100., 341300., 342200.])