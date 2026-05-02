# Cells 7 to 

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

# Transformer Model Building
embedding_dim = 32  # We can Adjust the embedding dimension based on our preference
num_heads = 2
ff_dim = 32

inputs = Input(shape=(1000,))
embedding_layer = Embedding(input_dim=10000, output_dim=embedding_dim)(inputs)
transformer_block = embedding_layer

for _ in range(num_heads):
    attn_output = Attention(use_scale=True)([transformer_block, transformer_block])
    transformer_block = LayerNormalization(epsilon=1e-6)(transformer_block + attn_output)
    transformer_block = Dropout(0.1)(transformer_block)

transformer_block = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(transformer_block)
transformer_block = tf.keras.layers.GlobalAveragePooling1D()(transformer_block)
transformer_block = Dropout(0.1)(transformer_block)
transformer_block = Dense(20, activation='relu')(transformer_block)
output_layer = Dense(2, activation='softmax')(transformer_block)

model = Model(inputs=inputs, outputs=output_layer)