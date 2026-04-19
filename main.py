#Para correr la aplicación se debe hacer el siguiente comando
#uvicorn main:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from train_model import deploy_model
import os
import kagglehub
import pandas as pd

# Agregando los datos nuevos -----------------------------------------
path = kagglehub.dataset_download("sbhatti/financial-sentiment-analysis")
data = pd.read_csv(path + "/data.csv")
# --------------------------------------------------------------------

if os.path.exists('model.pkl'):
    print("MODEL FOUND!")
else:
    train_X = data['Sentence']
    train_y = data['Sentiment']

    # Volviendo las clasificaciones numéricas ------------------------------
    for r in range(len(train_y)):
        if train_y[r] == 'positive':
            train_y[r] = 2
        elif train_y[r] == 'negative':
            train_y[r] = 0
        else:
            train_y[r] = 1
    #------------------------------------------------------------------------

    deploy_model(train_X, train_y.astype(int))

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputData): ## Equivalente a su score.py
    
    X = vectorizer.transform([data.text])

    # Predict
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }