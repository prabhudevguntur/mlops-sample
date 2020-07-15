import numpy as np
from code.training.score import train_model, feature_eng


def test_score_model():
    
    # Arrange
    data = pd.read_csv('sample.csv').drop("median_house_value", axis=1)
    data = {'data':data.values.tolist()}
    init()
   
    # Act
    preds = run(data, {})
    
    # Assert
    preds =  preds["result"]
    np.testing.assert_almost_equal(preds, [452600., 358500., 352100., 341300., 342200.])