from fastapi import FastAPI, HTTPException
import pickle
import numpy as np

from src.data_model import ModelFeatures


filename = 'data/model/xgb_model_top_10.sav'
model = pickle.load(open(filename, 'rb'))

# with open(filename, "rb") as f:
#     model = pickle.load(f)

app = FastAPI()


@app.get("/api/test")
async def test():
    return "Hello World!"

@app.get("/api/test_predict")
async def test():
    test_data = {
        'Heidelberg': 86.93,
        'Essen': 0.0,
        'Dortmund': 0.0,
        'Ulm': 801.76,
        'Bergedorf': 0.0,
        'Wuerzburg': 9.89,
        'Bottrop': 4.1721,
        'Darmstadt': 828.59,
        'Goettingen': 0.0,
        'Recklinghausen': 0.0
    }
    input_data = np.array(list(ModelFeatures(**test_data).model_dump().values())).reshape(1, -1)
    
    # Perform prediction using the loaded model
    predicted_installs = model.predict(input_data)
    
    return {
        'predicted_installs': predicted_installs.tolist()
    }
   
@app.get('/api/predict_v2')
async def predict():
    features = {
        "Heidelberg": 0.0,
        "Essen": 0.0,
        "Dortmund": 0.0,
        "Ulm": 0.0,
        "Bergedorf": 0.0,
        "Wuerzburg": 0.0,
        "Bottrop": 0.0,
        "Darmstadt": 0.0,
        "Goettingen": 0.0,
        "Recklinghausen": 0.0
    }
    try:
        input_data = np.array(list(features.values())).reshape(1, -1)
        input_data = np.array(list(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)).reshape(1, -1)
        predicted_installs = model.predict(input_data)
        
        return {
            'predicted_installs': predicted_installs.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post('/api/predict')
async def predict(features: ModelFeatures):
    try:
        input_data = np.array(list(features.model_dump().values())).reshape(1, -1)
        predicted_installs = model.predict(input_data)
        
        return {
            'predicted_installs': predicted_installs.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))