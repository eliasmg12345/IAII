import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
# IMPORTAR DATOS DEL DATASET
from sklearn import datasets
dataset = datasets.load_breast_cancer()
print('Información en el dataset:')
print(dataset.keys())
# SE OBTIENE LAS CARACTERÍSTICAS DEL DATASET
print('Características del dataset:')
print(dataset.DESCR)
#Se seleccionan todas las columnas de entrada
x = dataset.data
#Variables de entrada
entrada=dataset.feature_names
#CORRIGIENDO VALORES DE ENTRADA para que coincidan con dfquery
entrada=[u.replace('mean ','') for u in entrada]
entrada=[u.replace(' error','') for u in entrada]
entrada=[u.replace('worst ','') for u in entrada]
entrada=[u.replace('dimension','dimensión') for u in entrada]
entrada=[u.replace('area','área') for u in entrada]
#Se selecciona la columna de salida
y = dataset.target
#variables de salida
salidas=dataset.target_names
# SELECCIONAR DATOS DE train y test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size= 0.2)
# NORMALIZAR LOS DATOS
from sklearn.preprocessing import StandardScaler
escalar = StandardScaler()
xtrain = escalar.fit_transform(xtrain)
xtest = escalar.transform(xtest)
# CREAR EL MODELO DE RED NEURONAL
from sklearn.neural_network import MLPClassifier
modelo=MLPClassifier(activation='logistic', max_iter=100,hidden_layer_sizes=(3, ),solver='lbfgs')
# ENTRENAR EL MODELO CON LOS DATOS DE train
modelo.fit(xtrain, ytrain)
modelo.predict(xtrain)
# PRUEBA (PREDICCIÓN) CON LOS DATOS DE test
ypred = modelo.predict(xtest)
print('Exactitud con datos test:',modelo.score(xtest, ytest))
print('Diferencia entre y-pred (predicción) Y y_test (dato)')
print(ypred - ytest)
#Matriz de Confusión para verificar el rendimiento del modelo
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(ytest, ypred)
print('Matriz de Confusión:',matriz)
# DATOS DE CONSULTA (query)
dfquery = pd.DataFrame({"radius":[28],"texture":[39],
"perimeter":[188],"área":[2500],"smoothness":[0.1],
"compactness":[0.3],"concavity":[0.4],
"concave points":[0.2],"symmetry":[0.3],"fractal dimensión":[0.09],"radius":[2.8],"texture":[4.8],
"perimeter":[21.98],"área":[542.2],"smoothness":[0.03],
"compactness":[0.135],"concavity":[0.39],
"concave points":[0.05],"symmetry":[0.07],
"fractal dimensión":[0.03],"radius":[36],"texture":[49.5],
"perimeter":[251],"área":[4254],"smoothness":[0.22],
"compactness":[1],"concavity":[1.2],"concave points":[0.2],
"symmetry":[0.6],"fractal dimensión":[0.2] } )
xq = dfquery[entrada]
print("datos de consulta: ",xq)
# NORMALIZAR LOS DATOS DE CONSULTA
xq = escalar.transform(xq)
# RESULTADO DE LA CONSULTA
print('predicciones:', modelo.predict(xq))