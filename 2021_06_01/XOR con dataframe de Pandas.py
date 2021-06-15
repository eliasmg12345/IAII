import numpy as np
import pandas as pd
df = pd.DataFrame({
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "y": [0, 1, 1, 0] } )
entrada=["x1", "x2"]
salida=["y"]
X_train = df[entrada]
y_train = df[salida]
if len(salida)==1:
    y_train=np.ravel(y_train)
from sklearn.neural_network import MLPClassifier
modelo=MLPClassifier(activation='logistic', max_iter=250,
                     hidden_layer_sizes=(3, ),solver='lbfgs')
modelo.fit(X_train, y_train)
modelo.predict(X_train)
print('precisión: ',modelo.score(X_train, y_train))
print('predicciones: ',modelo.predict(X_train))