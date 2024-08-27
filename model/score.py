import json
import joblib
import numpy as np
import os

# Initialize the model
def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

# Run the model
def run(data):
    try:
        # Parse the input data from JSON
        data = json.loads(data)
        input_data = np.array(data['data'])
        
        # Make predictions
        predictions = model.predict(input_data)
        
        # Return the predictions as JSON
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        return json.dumps({"errorrrr": str(e)})

