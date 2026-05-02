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

import kagglehub
path = kagglehub.dataset_download("haisemei/fact-checking-dataset-label")
file = path + '\HIGH_CONFIDENCE_train_dataset.json'
data = pd.read_json(file)


# Encode label
def encode_label(row):
    if row['label'] == 'SUPPORTED':
        return 1
    elif row['label'] == 'NEI':
        return 0
    elif row['label'] == 'REFUTED':
        return 0

# Acá se prepara el texto concatenando claim y contexts con separadores
def concat_claim_evidence(row):
    return '[CLS] Claim: ' + row['claim'] + " [SEP] Context: " + " ".join(row['contexts'])

data['label'] = data.apply(encode_label, axis = 1)
data['fact'] = data.apply(concat_claim_evidence, axis = 1)
df = data.copy()


# df['cleaned_sentence'] = df['Sentence'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
# df['cleaned_sentence'] = df['cleaned_sentence'].apply(lambda x: x.lower())
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))
# df['cleaned_sentence'] = df['cleaned_sentence'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))


X_train, X_test, y_train, y_test = train_test_split(df['fact'], df['label'], test_size=0.9, random_state=42)

# Blanceo de clases
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42, replacement=False, sampling_strategy='not minority')
X_train, y_train = rus.fit_resample(X_train.to_frame(), y_train.to_frame())
# Turn train data into pd.Series again...
y_train = y_train['label']
X_train = X_train['fact']


# Label Encoding for Label Column
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_sequences, maxlen=1000, truncating='post', padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=1000, truncating='post', padding='post')