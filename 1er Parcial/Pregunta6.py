
#determinar si la postulante es aceptada en el equipo de voley en funcion de su estatura
import numpy as np
#preparar los datos
X=np.array([168,180,155,160,158,179,182,150,178,170,153,183]).reshape(-1,1)
y = np.array([1,1,0,0,0,1,1,0,1,0,0,1])
#Entrenar la regresion logistica. El modelo aprende los coeficientes minimizan la funcion del costo
#importar la clase LogisticRegresion de scikit-learn
from sklearn.linear_model import LogisticRegression
#crear una instancia de la regresion logistica
regresion_logis=LogisticRegression()
#entrenar la regresion logistics con datos del train
regresion_logis.fit(X,y)
#hacendo predicciones con las medidas de 151 y 172 centimetros
X_new=np.array([151,172]).reshape(-1,1)
#usar el modelo entrenado para obtener predicciones con datos nuevos
prediccion=regresion_logis.predict(X_new)
print('Aceptada al equipo (1) o no aceptada (0) _estaturas de 151 y 172 cm')
print(prediccion)
