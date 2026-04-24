import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import kagglehub
import pandas as pd
import numpy as np

# Agregando los datos
path = kagglehub.dataset_download("sbhatti/financial-sentiment-analysis")
data = pd.read_csv(path + "/data.csv") # Pongo este límite para que no se me acabe la memoria

# Encode sentiment
def encode_sentiment(row):
    if row['Sentiment'] == 'positive':
        return 2
    elif row['Sentiment'] == 'neutral':
        return 1
    elif row['Sentiment'] == 'negative':
        return 0
data['sentiment'] = data.apply(encode_sentiment, axis = 1)
df = data.copy()



df['cleaned_sentence'] = df['Sentence'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
df['cleaned_sentence'] = df['cleaned_sentence'].apply(lambda x: x.lower())
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df['cleaned_sentence'] = df['cleaned_sentence'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))


X_train, X_test, y_train, y_test = train_test_split(df['cleaned_sentence'], df['sentiment'], test_size=0.9, random_state=42)

# Blanceo de clases
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42, replacement=False, sampling_strategy='not minority')
X_train, y_train = rus.fit_resample(X_train.to_frame(), y_train.to_frame())
# Turn train data into pd.Series again...
y_train = y_train['sentiment']
X_train = X_train['cleaned_sentence']


# Label Encoding for Sentiment Column
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_sequences, maxlen=100, truncating='post', padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=100, truncating='post', padding='post')