import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

X_p = np.array([0.75, 1, 1.25, 1.5, 1.65, 1.75, 2, 2.25, 2.5, 2.75,3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.25]).reshape(-1,1)
y_p = np.array([10, 15, 17, 20, 22, 24, 30, 35, 42, 48, 56,62, 68, 73, 78, 80, 83, 85, 89])

plt.scatter(X_p,y_p)
plt.show()

from sklearn.model_selection import train_test_split

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.2)

from sklearn.preprocessing import PolynomialFeatures

poli_reg = PolynomialFeatures(degree=3)

X_train_poli = poli_reg.fit_transform(X_train_p)
X_test_poli = poli_reg.fit_transform(X_test_p)

pr = linear_model.LinearRegression()
pr.fit(X_train_poli, y_train_p)
#una prediccion
Y_pred_pr = pr.predict(X_test_poli)

plt.scatter(X_test_p,y_test_p)
plt.plot(X_test_p,Y_pred_pr,color='red',linewidth=3)
plt.show()

print()
print('Datos de la regreion polinomial')
print()
print('valor pendiente b')
print(pr.coef_)

print('valor de interseccion a')
print(pr.intercept_)

print('Precision')
print(pr.score(X_train_poli, y_train_p))

