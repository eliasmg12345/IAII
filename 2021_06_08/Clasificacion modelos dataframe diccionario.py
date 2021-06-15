import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
#DATOS
df=pd.DataFrame({
    "peso":[55,60,62,67,65,75,64,77,78,70,72,65],
    "estatura":[168,180,155,160,179,155,178,170,160,185,187,160],
    "desicion":[1,1,0,0,1,0,1,0,0,1,1,0]
})
entrada=["peso","estatura"]
salida=["desicion"]
x=df[entrada]
y=df[salida]
if len(salida)==1:
    y=np.ravel(y)
#SELECCIONAR DATOS DE TRAIN Y TEST PARA ENTRENAMIENTO Y PRUEBA
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
#NORMALIZAR LOS DATOS
from sklearn.preprocessing import StandardScaler
escalar=StandardScaler()
xtrain=escalar.fit_transform(xtrain)
xtest=escalar.transform(xtest)
#TRAIN
modelo=MLPClassifier(activation='logistic',max_iter=100,
                     hidden_layer_sizes=(3,),solver='lbfgs')
modelo.fit(xtrain,ytrain)
modelo.predict(xtrain)
#TEST
modelo.predict(xtest)
print("presicion: ",modelo.score(xtest,ytest))
#DATOS DE CONSULTA (query)
dfquery=pd.DataFrame({
    "peso":[61,60],
    "estatura":[151,170]
})
xq=dfquery[entrada]
print("datos de cosnsulta: ",xq)
#NORMALIZAR LOS DATOS DE CONSULTA
xq=escalar.transform(xq)
#RESULTADO DE LA COSNULTA
print('predicciones: ',modelo.predict(xq))