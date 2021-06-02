import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
def polinomial():
    #Generar 100 datos que no tienen una distribución lineal
    np.random.seed(0)
    X = 10 * np.random.rand(100, 1) - 5
    y = 0.5 * X ** 2 + 1.5 * X + 3 + np.random.randn(100, 1)
    # Graficar los datos
    fig = plt.figure(figsize = (8, 6))
    plt.plot(X,y,linestyle=' ',marker='o');
    # Importar la clase PolynomialFeatures e instanciarla:
    from sklearn.preprocessing import PolynomialFeatures
    # Generar un polinomio de grado 2
    poly = PolynomialFeatures(2)
# Transformar los datos
    print('Datos originales (5): ',X[:5])
    X_poly = poly.fit_transform(X)
    print('Datos transformados (5): ',X_poly[:5])
    # Instanciar un modelo de regresión lineal y entrenarlo
    model = LinearRegression()
    model.fit(X_poly, y)
    LinearRegression(copy_X=True, fit_intercept=True,n_jobs=None, normalize=False)
    print('Coeficientes del polinomio de regresión: ',model.intercept_, model.coef_)
    # Predicción y gráfico
    prediction = model.predict(X_poly)
    fig = plt.figure(figsize = (8, 6))
    plt.plot(X, y, linestyle=' ', marker='o');
    yp = prediction.reshape(-1, )
    plt.plot(X, yp, linestyle=' ', marker='*', color ='red');
    plt.show()