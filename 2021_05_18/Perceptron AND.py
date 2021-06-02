from math import fabs
from numpy import array, dot, random
import pylab as plt
def aprendizaje():
    # Función de transferencia escalón:
    A= 1
    B= int(input("A=1, B<0 o -1> = "))
    escalon = lambda NET: B if NET < 0 else A
    # datos para train, entradas x1, x2, x0 (x0 siempre 1)
    # salida deseada d: (array[x1, x2, x0], d)
    train_data = [
        (array([0,0,1]), 0),
        (array([0,1,1]), 0),
        (array([1,0,1]), 0),
        (array([1,1,1]), 1) ]
    # Número de patrones de entrada
    npe = len(train_data)
    n=len(train_data[0][0])
    # pesos iniciales aleatorios: w0, w1, w2
    w = random.rand(n)
    # tasa de aprendizaje (alfa)
    alfa = 0.2
    # número máximo de iteraciones de aprendizaje
    itmax = 100
    # error máximo
    emax = 0.01
    # entrenamiento (calcula valores de los pesos)
    # iteraciones de 0 a itmax-1
    for t in range(itmax):
        # lista de valores de error, inicializada
        errores = []
        # se trabaja con cada patrón de entrada
        for p in range(npe):
            # se selecciona el patrón de entrada p de train
            x, d = train_data[p]
            # entrada neta net, producto escalar de w, x
            net = dot(w, x)
            # calcula la salida obtenida
            y = escalon(net)
            # error=salida deseada(d)-salida obtenida(y)
            error = d - y
            errores.append(error)
            # actualiza pesos (regla de aprendizaje)
            w = w + alfa * error * x
        # Ve error<emax para todos los patrones de entrada
        print("errores de Train: ", errores)
        sw = 0
        for p in range(npe):
            if fabs(errores[p]) > emax:
                sw = 1
                print("ERROR EN TRAIN NO APRENDÍO")
    if sw == 0:
        # datos para test: (array[x1, x2, x0], d) con x0=1
        test_data = [
            (array([0.02, 0.01, 1]), 0),
            (array([0.98, 0.01, 1]), 0)]
        # calcula las salidas para datos de test y el error
        errores = []
        for x, d in test_data:
            net = dot(x, w)
            y = escalon(net)
            error = d - y
            errores.append(error)
        print("errores de Test: ", errores)
        ndt = len(errores)
        for p in range(ndt):
            if fabs(errores[p]) > emax:
                sw = 1
                print("ERROR EN TEST NO APRENDÍO")
    if sw == 0:
        print("APRENDIÓ")
        return w
        print(w)
    else:
        print("no aprendió en ", itmax,"iteraciones")

# calcula y muestra las salidas para una consulta
def query(w):
    query_data = [
        (array([0.01, 0.99, 1])),
        (array([0.99, 0.98, 1])),]
    # Función de transferencia escalón:
    A = 1
    B = int(input("A=1, B<0 o -1> = "))
    escalon = lambda NET: B if NET < 0 else A
    print("resultado de la consulta")
    for xc in query_data:
        net = dot(xc, w)
        print("{} -> {}".format(xc[:2], escalon(net)))