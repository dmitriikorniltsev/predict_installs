from fastapi import FastAPI, HTTPException
import pickle
import numpy as np

from src.data_model import ModelFeatures


filename = 'data/model/xgb_model_top_10.sav'

with open(filename, "rb") as f:
    model = pickle.load(f)

app = FastAPI()


@app.get("/api/test")
async def test():
    return "Hello World!"
    
# @app.post('/api/predict')
@app.get('/api/predict')
async def predict(features: ModelFeatures):
    try:
        input_data = np.array(list(features.model_dump().values())).reshape(1, -1)
        
        predicted_installs = model.predict(input_data)
        
        return {
            'predicted_installs': predicted_installs.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))