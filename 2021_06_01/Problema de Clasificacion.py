import numpy as np
import pandas as pd
df= pd.DataFrame({
    "x1": [0.3, 0.1, 0.4, 0.5],
    "x2": [0.7, 0.2, 0.3, 0.8],
    "y1": [0,0,0,1],
    "y2":[0,1,1,1] })
entrada=["x1", "x2"]
salida=["y1","y2"]
X_train = df[entrada]
y_train = df[salida]
if len(salida)==1:
    y_train=np.ravel(y_train)
from sklearn.neural_network import MLPClassifier
modelo = MLPClassifier((3, ), random_state = 0,
learning_rate_init = 0.1, activation = "logistic")
modelo.fit(X_train, y_train)
print('precisión:',modelo.score(X_train, y_train))
print('predicciones: ', modelo.predict(X_train))