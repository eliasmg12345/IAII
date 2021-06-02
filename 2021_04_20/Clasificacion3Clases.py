# Clasificador de regresión logística de 3 clases
# Longitud y ancho del sépalo de datos del iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
def log1():
    # importar dataset de iris
    iris = datasets.load_iris()
    # Tomar las 2 primeras características (columnas)
    X = iris.data[:, :2]
    Y = iris.target
    print('Datos:')
    print(X, Y)
    # Aplicar regresión logística
    logreg = LogisticRegression(C=1e5)
    # Crear una instancia del Logistic Regression Classifier y ajustar (train) los datos.
    logreg.fit(X, Y)
    # Dibujar el perímetro de decision.
    # Asignar un color a cada punto en la malla: [x_min, x_max]*[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02 # Tamaño de paso en el mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    # Colocar el resultado en un color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    # Dibujar los datos (puntos) de entrenamiento
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()