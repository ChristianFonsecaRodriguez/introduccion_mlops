from model import IrisModel, IrisSpecies
from fastapi import FastAPI
import uvicorn
import joblib
import json

app = FastAPI()
model = IrisModel()

@app.get('/train')
def train_species():
    model.model = model.train_model()
    joblib.dump(model.model, model.model_name)
    return "El modelo ha sido entrenado y guardado."

@app.post('/predict')
def predict_species(iris: IrisSpecies):
    data = iris.dict()
    prediction, probability = model.predict_species(
        data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']
    )
    return {
        'prediction': prediction,
        'probability': probability
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

