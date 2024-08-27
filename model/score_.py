import json
import joblib
import numpy as np
import os

def init():
    global model
    
    # Load the model from the file path
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

def run(data):
    try:
        # Parse the input data
        input_data = json.loads(data)
        # Assuming the input is a list like [4, 3, 1940, 10500, 1, 0, 1140, 1976, 1992, 'Redmond']
        
        # Convert input data to numpy array or DataFrame as needed
        input_array = np.array([input_data])
        
        # Perform prediction using the model
        prediction = model.predict(input_array)
        
        # Return the prediction
        return json.dumps({"prediction": prediction.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
