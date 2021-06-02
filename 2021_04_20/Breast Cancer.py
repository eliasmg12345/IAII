# Se importan la librerias a utilizar
from sklearn import datasets
# Se importa los datos de la librería de scikit-learn
dataset = datasets.load_breast_cancer()
# print(dataset)
# Se verifica la información contenida en el dataset
print('Información en el dataset:')
print(dataset.keys())
print()
# Se obtiene las características del dataset
print('Características del dataset:')
print(dataset.DESCR)
# Se seleccionan todas las columnas de entrada
X = dataset.data
# Se selecciona la columna de salida
y = dataset.target
# Se implementa la Regresión Logística
from sklearn.model_selection import train_test_split
# Se seleccionan datos de train y test para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#Se escalan los datos
from sklearn.preprocessing import StandardScaler
escalar = StandardScaler()
X_train = escalar.fit_transform(X_train)
X_test = escalar.transform(X_test)
# Se define el algoritmo a utilizar
from sklearn.linear_model import LogisticRegression
algoritmo = LogisticRegression()
# Entrenamiento del modelo con los datos de train
algoritmo.fit(X_train, y_train)
# Prueba (predicción) con los datos de test
y_pred = algoritmo.predict(X_test)
print('Diferencia entre y-pred (predicción) Y y_test (dato)')
print(y_pred - y_test)
#Matriz de Confusión para verificar el rendimiento del modelo
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(matriz)
# CÁLCULOS PARA DATOS DESBALANCEADOS
# Calculo la precisión del modelo
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precisión del modelo:')
print(precision)
# Calculo la exactitud del modelo
from sklearn.metrics import accuracy_score
exactitud = accuracy_score(y_test, y_pred)
print('Exactitud del modelo:')
print(exactitud)
# Calculo la sensibilidad del modelo
from sklearn.metrics import recall_score
sensibilidad = recall_score(y_test, y_pred)
print('Sensibilidad del modelo:')
print(sensibilidad)