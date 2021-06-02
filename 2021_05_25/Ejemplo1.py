from math import fabs
from numpy import array, dot, random
import pylab as plt
def aprendizaje():
    # Función de transferencia escalón:
    A= 1
    B= int(input("A=1,B<0 o -1> = "))
    escalon = lambda x: B if x < 0 else A
    # datos para train, entradas x1, x2, x0 (x0 siempre 1)
    # salida deseada d: (array([x1, x2, x0]), d)
    train_data = [
        (array([0.55,1.68,1]), 1),
        (array([0.60,1.80,1]), 1),
        (array([0.62,1.55,1]), 0),
        (array([0.67,1.60,1]), 0),
        (array([0.65,1.79,1]), 1),
        (array([0.75,1.55,1]), 0),
        (array([0.64,1.78,1]), 1),
        (array([0.77,1.70,1]), 0),
        (array([0.78,1.60,1]), 0),
        (array([0.70,1.85,1]), 1) ]
    # Número de patrones de entrada
    npe = len(train_data)
    # Número de variables de entrada
    n = len(train_data[0][0]) - 1
    # pesos iniciales aleatorios: w0, w1, w2
    w = random.rand(n+1)
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
            # selecciona el patrón de entrada p de train
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
        print("errores de Train: ",errores)
        sw=0
        for p in range(npe):
            if fabs(errores[p])>emax:
                sw=1
                print("ERROR EN TRAIN")
    if sw==0:
        # realiza el TEST si ha aprendido en TRAIN
        # datos para test: (array[x1, x2, x0], d) con x0=1
        test_data = [
            (array([0.72,1.87,1]), 1),
            (array([0.65,1.60,1]), 0) ]
        # calcula las salidas para datos de test y el error
        errores=[]
        for x, d in test_data:
            net = dot(x, w)
            y= escalon(net)
            error = d - y
            errores.append(error)
            print("errores de Test: ",errores)
            ndt = len(errores)
            for p in range(ndt):
                if fabs(errores[p])>emax:
                    sw=1
                print("ERROR EN TEST")
        if sw==0:
            print(w)
            print("APRENDIÓ")
            return w
        else:
            print("NO APRENDIÓ")

# calcula y muestra las salidas para una CONSULTA
def consulta(w):
    query_data = [
        (array([0.61,1.51,1])),
        (array([0.60,1.75,1])) ]
    # Función de transferencia escalón:
    A= 1
    B= int(input("B<0 o -1> = "))
    escalon = lambda x: B if x < 0 else A
    print("resultado de la consulta")
    for xc in query_data:
        net = dot(xc, w)
        print("{} -> {}".format(xc[:2], escalon(net)))