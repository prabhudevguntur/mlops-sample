import json
import numpy as np
from azureml.core.model import Model
import joblib

def init():
    global model
    global full_data_pipeline
    model_path = Model.get_model_path(
        model_name="rf_tuned_model.pkl")
    pipeline_path = Model.get_model_path(
        model_name="full_pipeline.pkl")
    model = joblib.load(model_path)
    full_data_pipeline = joblib.load(pipeline_path)

def run(raw_data, request_headers):
    data = json.loads(raw_data)["data"]
    data = full_data_pipeline.transform(np.array(data))
    result = model.predict(data)
    return {"result": result.tolist()}

init()
test_row = {"data":[-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252, 'NEAR BAY']}
request_header = {}
prediction = run(test_row, request_header)
print("Test result: ", prediction)