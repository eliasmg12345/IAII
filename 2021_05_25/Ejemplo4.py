from math import fabs
import numpy as np
from numpy import array, dot, random
def aprendizaje():
    # Función de transferencia escalón:
    A= 1
    B= int(input("A=1, B<0 o -1> = "))
    escalon = lambda NET: B if NET < 0 else A
    # datos para train, entradas x1, x2, ... , x0
    # (x0 siempre es 1). Salidas deseadas d1, d2, ... :
    # (array[x1, x2, ..., x0], array[d1, d2, ...])
    # (array[x1, x2, ..., x0], array[d1, d2, ...])
    train_data = [
        (array([0.55,1.77,1]), array([1,1])),
        (array([0.60,1.80,1]), array([1,1])),
        (array([0.65,1.55,1]), array([0,0])),
        (array([0.45,1.60,1]), array([1,0])),
        (array([0.65,1.79,1]), array([1,1])),
        (array([0.75,1.55,1]), array([0,0])),
        (array([0.64,1.78,1]), array([1,1])),
        (array([0.90,1.79,1]), array([0,1])),
        (array([0.78,1.60,1]), array([0,0])),
        (array([0.68,1.85,1]), array([1,1])) ]
    # Número de variables de entrada
    n = len(train_data[0][0]) - 1
    # Número de variables de salida
    m = len(train_data[0][1])
    # Número de patrones de entrada
    npe = len(train_data)
    # pesos iniciales aleatorios:
    # w00,w01, ...,w0m,w10,w11, ...,w1m,wn0,...,wnm
    w=[]
    for k in range(m):
        w.append(array(random.rand(n+1)))
    print(w)
    # tasa de aprendizaje (alfa)
    alfa = 0.2
    # número máximo de iteraciones de aprendizaje
    itmax = 50
    # error máximo
    emax = 0.01
    # entrenamiento (calcula valores de los pesos)
    # iteraciones de 0 a itmax-1
    for t in range(itmax):
        # lista de valores de error, inicializada
        errores = np.zeros((npe,m))
        # se trabaja con cada patrón de entrada
        for p in range(npe):
            # selecciona el patrón de entrada p de train
            x, d = train_data[p]
            # entrada neta net, producto escalar de w, x
            for s in range(m):
                net = dot(w[s], x)
                # calcula la salida obtenida
                y = escalon(net)
                # calcula el error
                # = salida deseada(d) - salida obtenida(y)
                error = d[s] - y
                errores[p][s]=error
                # actualiza pesos de conexiones
                # (regla de aprendizaje)
                if error!=0:
                    w[s] = w[s] + alfa * error * x
        # Ve error<emax para todos patrones entrada
        print("errores de Train: ",errores)
        sw=0
        for p in range(npe):
            for s in range(m):
                if fabs(errores[p][s])>emax:
                    sw=1
                    print("ERROR EN TRAIN")
    if sw==0:
        # datos para test: (array[x1, x2, x0], d) con x0=1
        test_data = [
            (array([0.70,1.87,1]), array([1,1])),
            (array([0.68,1.60,1]), array([0,0])) ]
        # número de patrones de entrada de test
        npe = len(test_data)
        # calcula salidas para datos de test y el error
        errores = np.zeros((npe,m))
        # se trabaja con cada patrón de entrada de test
        for p in range(npe):
            # se selecciona el patrón de entrada p de test
            x, d = test_data[p]
            # entrada neta net, producto escalar de w, x
            for s in range(m):
                net = dot(w[s], x)
                # calcula la salida obtenida
                y = escalon(net)
                # calcula el error=
                #salida deseada(d) - salida obtenida(y)
                error = d[s] - y
                errores[p][s]=error
        # Ve error<emax para todos los patrones de entrada de test
        print("errores de Test: ",errores)
        sw=0
        for p in range(npe):
            for s in range(m):
                if fabs(errores[p][s])>emax:
                    sw=1
                    print("ERROR EN TEST")
    if sw==0:
        print("APRENDIÓ")
        return w
    else:
        print("NO APRENDIÓ")

# calcula y muestra las salidas para una consulta
def query(w):
    query_data = [
        (array([0.61,1.51,1])),
        (array([0.60,1.79,1])) ]
    # Función de transferencia escalón:
    A= 1
    B= int(input("A=1, B<0 o -1> = "))
    escalon = lambda x: B if x < 0 else A
    # número de patrones de entrada de CONSULTA
    npe = len(query_data)
    # Número de variables de entrada
    n = len(query_data[0]) - 1
    # Número de variables de salida
    m = int(input("Número de variables de salida= "))
    print("resultado de la consulta")
    # inicializar el vector de salidas en cero
    y=np.zeros((npe,m))
    # trabaja con cada patrón de entrada de consulta
    for p in range(npe):
        # selecciona el patrón de entrada p de consulta
        x= query_data[p]
        # entrada neta net, producto escalar de w, x
        for s in range(m):
            net = dot(w[s], x)
            # calcula la salida obtenida
            y[p][s]= escalon(net)
            # muestra patron de entrada y la salida obtenida
            print("{} -> {}".format(x[:n], y[p]))