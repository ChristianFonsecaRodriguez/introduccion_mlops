import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from pydantic import BaseModel

class IrisSpecies(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class IrisModel(object):
    def __init__(self):
        self.df = pd.read_csv('iris.csv')
        self.model_name = 'iris_model.pkl'
        try:
            self.model = joblib.load(self.model_name)
        except:
            print('Se debe entrenar al modelo primero.')

    def train_model(self):
        X = self.df.drop('species', axis=1)
        y = self.df['species']

        clf = RandomForestClassifier()
        self.model = clf.fit(X, y)
        return self.model


    def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
        data_in = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = self.model.predict(data_in)
        probability = self.model.predict_proba(data_in).max()
        return prediction, probability

