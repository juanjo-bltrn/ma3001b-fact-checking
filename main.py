#Para correr la aplicación se debe hacer el siguiente comando
#uvicorn main:app --reload

from fastapi import FastAPI
from pydantic import BaseModel, json
import pickle
import os
import kagglehub
import pandas as pd
import keras
from kerasTransformer.model_train import *
import ast

if os.path.exists('model.keras'):
    print("MODEL FOUND!")
else:
    deploy_model()

app = FastAPI()

with open("model.keras", "rb") as f:
    model = keras.saving.load_model("model.keras")

class InputData(BaseModel):
    claim: str
    contexts: list[str]

@app.post("/predict")
def predict(comment: InputData): ## Equivalente a su score.py
    #comment = comment.text
    #print( "Received Comment:", comment)  # Debugging line to check the input format
    # User Input Prediction
    #comment = ast.literal_eval(comment)
    cleaned_comment = '[CLS] Claim: ' + comment.claim + " [SEP] Context: " + " ".join(comment.contexts)
    print("Cleaned Comment:", cleaned_comment)

    sequence = tokenizer.texts_to_sequences([cleaned_comment])
    padded_sequence = pad_sequences(sequence, maxlen=100, truncating='post', padding='post')
    prediction = model.predict(padded_sequence)
    prediction = prediction[0]

    pred = prediction.argmax()
    pred = int(pred)
    if pred == 0:
        pred_label = 'REFUTED'
    else:
        pred_label = 'SUPPORTED'
    prob = prediction.max()

    prediction = {
        'predicted_label': pred_label
    }

    return prediction