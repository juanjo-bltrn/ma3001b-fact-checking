#Para correr la aplicación se debe hacer el siguiente comando
#uvicorn main:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
import kagglehub
import pandas as pd
import keras
from kerasTransformer.model_train import *

# Agregando los datos nuevos -----------------------------------------
path = kagglehub.dataset_download("sbhatti/financial-sentiment-analysis")
data = pd.read_csv(path + "/data.csv")
# --------------------------------------------------------------------

if os.path.exists('model.keras'):
    print("MODEL FOUND!")
else:
    deploy_model()

app = FastAPI()

with open("model.keras", "rb") as f:
    model = keras.saving.load_model("model.keras")

class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict(comment: InputData): ## Equivalente a su score.py
    comment = comment.text
    # User Input Prediction
    cleaned_comment = re.sub(r'[^a-zA-Z\s]', '', comment.lower())
    cleaned_comment = ' '.join(word for word in cleaned_comment.split() if word not in stop_words)
    sequence = tokenizer.texts_to_sequences([cleaned_comment])
    padded_sequence = pad_sequences(sequence, maxlen=100, truncating='post', padding='post')
    prediction = model.predict(padded_sequence)
    prediction = prediction[0]

    pred = prediction.argmax()
    prob = prediction.max()

    prediction = {
        'prediction': int(pred),
        'probability': float(prob)
    }

    return prediction