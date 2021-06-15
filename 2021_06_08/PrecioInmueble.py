import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
# DATOS EN EXCEL
df = pd.read_excel('inmueble.xlsx', sheet_name= 'Hoja1')
print(df)
# VARIABLES DE ENTRADA
entrada=["calefaccion","garaje","dormitorios","banho","otroscuartos","terraza","chalet"]
# VARIABLES DE SALIDA
salida=["precio"]
# DATOS DE ENTRADA
x = df[entrada]
# DATOS DE SALIDA
y = df[salida]
# NORMALIZAR LAS SALIDAS
a=np.min(y)
b=np.max(y)
y=(0.9-0.1)/(b-a)*(y-a) + 0.1
if len(salida)==1:
    y=np.ravel(y)
# SELECCIONAR DATOS DE train y test PARA ENTRENAMIENTO Y PRUEBA
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.2)
print(xtest, ytest)
# NORMALIZAR LOS DATOS
from sklearn.preprocessing import StandardScaler
escalar = StandardScaler()
xtrain = escalar.fit_transform(xtrain)
xtest = escalar.transform(xtest)
# TRAIN
modelo=MLPRegressor(activation='logistic', max_iter=2000,hidden_layer_sizes=(20, 12),solver='lbfgs')
modelo.fit(xtrain, ytrain)
modelo.predict(xtrain)
# TEST
ypred=modelo.predict(xtest)
print("salida de test - salida de predicción: ", ytest - ypred)
print("precisión del test: ",modelo.score(xtest, ytest))
# DATOS DE CONSULTA (query)
dfquery = pd.DataFrame( {
"calefaccion":[1],"garaje":[1],"dormitorios":[3],"banho":[3],
"otroscuartos":[3], "terraza":[1],"chalet":[0] } )
xq = dfquery[entrada]
print("datos de consulta: ",xq)
# NORMALIZAR LOS DATOS DE CONSULTA
xq = escalar.transform(xq)
# RESULTADO DE LA CONSULTA
yq=modelo.predict(xq)
# DESNORMALIZAR LA SALIDA DE CONSULTA
yq=(b-a)/(0.9-0.1)*(yq-0.1) + a
print('predicciones:', yq)