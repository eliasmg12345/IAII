import matplotlib.pyplot as plt
import numpy as np
# importar herramientas para gráficos 3D
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, linear_model
# importar archivo de diabetes
diabetes = datasets.load_diabetes()
# Seleccionar datos para train y test
indices = (0, 1) # las 2 primeras columnas
X_train = diabetes.data[:-20, indices]
X_test = diabetes.data[-20:, indices]
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]
# Aplicar el modelo de regresión múltiple
ols = linear_model.LinearRegression()
# Ajusta la regresión con el gradiente descendente
ols.fit(X_train, y_train)
# Determinar si ha aprendido
# Con los datos de test se verifica si ha aprendido
print('aprendizaje 0=no, 1=si')
print(ols.score(X_test, y_test))
# Traza la figura
def plot_figs(fig_num, elev, azim, X_train, clf):
    # configura la figura
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    # dibuja los ejes (planos) cartesianos
    ax = Axes3D(fig, elev=elev, azim=azim)
    # dibuja los datos (puntos) X0, X1, y de train
    ax.scatter(X_train[:,0],X_train[:,1], y_train, c='k', marker='+')
    # dibuja la superficie (plano) de regresión
    ax.plot_surface(np.array([[-.1, -.1], [.15, .15]]), np.array([[-.1, .15], [-.1, .15]]), clf.predict(np.array([[-.1, -.1, .15, .15], [-.1, .15, -.1, .15]]).T).reshape((2, 2)), alpha=.5)
    # etiqueta los ejes cartesianos
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('Y')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
# Generar tres figuras desde distintas perspectivas
elev = 43.5
azim = -110
plot_figs(1, elev, azim, X_train, ols)
elev = -.5
azim = 0
plot_figs(2, elev, azim, X_train, ols)
elev = -.5
azim = 90
plot_figs(3, elev, azim, X_train, ols)
plt.show() # mostrar las figuras