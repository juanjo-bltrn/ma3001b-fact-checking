# Modificar el modelo para que sea un transformer pequeñito para que sea agnóstico al vocabulario

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def deploy_model(train_X, train_y):
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(train_X)

    model = LogisticRegression()
    model.fit(X_vec, train_y)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)