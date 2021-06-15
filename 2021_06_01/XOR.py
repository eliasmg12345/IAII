import numpy as np
import sklearn.neural_network
x = np.array([[0,0],[0,1],[1,0],[1,1]])
d = np.array([0,1,1,0])
modelo = sklearn.neural_network.MLPClassifier(
activation='logistic', max_iter=100,
hidden_layer_sizes=(3, ),solver='lbfgs')
modelo.fit(x, d)
modelo.predict(x)
print('precisión: ',modelo.score(x, d))
print('predicciones:', modelo.predict(x))