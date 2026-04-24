import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Attention, LayerNormalization, Dropout

from kerasTransformer.model import *
from kerasTransformer.data_processing import *

def deploy_model():
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_padded, y_train_encoded, epochs=100, validation_data=(X_test_padded, y_test_encoded))

    model.save('model.keras')


    y_pred_prob = model.predict(X_test_padded)

    y_pred = np.array([x.argmax() for x in y_pred_prob])

    print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
    print("Classification Report:\n", classification_report(y_test_encoded, y_pred))