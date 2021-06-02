def logistica():
    #determinar si un estudiante  aprueba un examen en funcion de las horas que ha estudiado
    import numpy as np
    #preparar los datos
    X=np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5]).reshape(-1,1)
    y = np.array([0, 0,0,0,0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    #Entrenar la regresion logistica. El modelo aprende los coeficientes minimizan la funcion del costo
    #importar la clase LogisticRegresion de scikit-learn
    from sklearn.linear_model import LogisticRegression
    #crear una instancia de la regresion logistica
    regresion_logis=LogisticRegression()
    #entrenar la regresion logistics con datos del train
    regresion_logis.fit(X,y)
    #hacer predicciones, si se estudia  1,2,3,4,5 o 6  horas
    X_new=np.array([1,2,3,4,5,6]).reshape(-1,1)
    #usar el modelo entrenado para obtener predicciones con datos nuevos
    prediccion=regresion_logis.predict(X_new)
    print('Aprobar(1) o no (0) estudiando 1,2,3,4,5,6 horas')
    print(prediccion)
    #se obtiene las probabilidades de la prediccion
    proba_predic=regresion_logis.predict_proba(X_new)
    print('probabilidad de reprobar  y aprobar estudiando 1,2,3,4,5,6 horas')
    print(proba_predic)
    print('probabilidad de aprobar estudiando 1,2,3,4,5,6 horas')
    print(proba_predic[:,1])