def logistica11():
    # Determinar si un estudiante aprueba un examen
    # en función de las horas que ha estudiado
    import numpy as np
    # preparar los datos
    X = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5]).reshape(-1,1)
    y = np.array([0, 0,0,0,0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    # Entrenar la regresión logística. El modelo aprende los coeficientes minimizan la función de costo
    # Importar la clase LogisticRegresion de scikit-learn
    from sklearn.linear_model import LogisticRegression
    # Crear una instancia de la Regresión Logística
    regresion_logistica = LogisticRegression()
    # Entrenar la regresión logística con datos de train
    regresion_logistica.fit(X, y)
    # Hacer predicciones, si se estudia 1, 2, 3, 4, 5 ó 6 horas
    X_nuevo = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    # Usar el modelo entrenado para obtener predicciones con datos nuevos
    prediccion = regresion_logistica.predict(X_nuevo)
    print('Aprobar (1) o no (0) estudiando 1,2,3, 4, 5 ó 6 horas ')
    print(prediccion)
    # Se obtienen las probabilidades de la predicción
    probabilidades_prediccion = regresion_logistica.predict_proba(X_nuevo)
    print('probabilidad de reprobar y de aprobar estudiando 1, 2, 3, 4, 5 ó 6horas: ')
    print(probabilidades_prediccion)
    print('probabilidad de aprobar estudiando 1, 2, 3, 4, 5, 6 horas ')
    print(probabilidades_prediccion[:, 1])