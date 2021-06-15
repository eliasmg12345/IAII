import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
#DATOS EN CSV
df=pd.read_csv('tiempo02.csv')
print(df)
#VARIABLES DE ENTRADA
entrada=["Year","Month"]
#VARIABLES DE SALIDA
salida=["Tmax"]
#DATOS DE NETRADA
x=df[entrada]
#DATOS DE SALIDA
y=df[salida]
if len(salida)==1:
    y=np.ravel(y)
#SELECCIONAR DATOS DE trainn y test PARA ENTRENAMIENTO Y PRUEBA
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
#NORMAILIZAR LOS DATOS
from sklearn.preprocessing import StandardScaler
escalar=StandardScaler()
xtrain=escalar.fit_transform(xtrain)
xtest=escalar.transform(xtest)
#TRAIN
modelo=MLPRegressor(hidden_layer_sizes=(20,20),
activation="relu",random_state=1,max_iter=2000,
solver='lbfgs')
modelo.fit(xtrain,ytrain)
modelo.predict(xtrain)
#TEST
ypred=modelo.predict(xtest)
print("salida de test - salida de predicción: ", ytest - ypred)
print("presicion del test ", modelo.score(xtest,ytest))
#DATOS DE CONSULTA (query)
dfquery=pd.DataFrame({
    "Year":[2019,2020],
    "Month":[5,1]
})
xq=dfquery[entrada]
print("datos de consulta: ",xq)
#NORMAILZIAR DATOS DE CONSULTA
xq=escalar.transform(xq)
#RESULTADO DE LA CONSULTA
print('predicciones : ',modelo.predict(xq))