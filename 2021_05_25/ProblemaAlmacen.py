from math import fabs
from numpy import array, dot, random
import pylab as plt
def aprendizaje():
    A= 1
    B= int(input("A=1,B<0 o -1> = "))
    escalon = lambda x: B if x < 0 else A
    #Ejemplos de aprendizaje tomados solamente seis 6
    train_data = [
        (array([10,200,1]), 1),
        (array([8,150,1]), 1),
        (array([7,170,1]), 1),
        (array([26,30,1]), 0),
        (array([24,32,1]), 0),
        (array([19,31,1]), 0)]
    npe = len(train_data)
    n = len(train_data[0][0]) - 1
    w = random.rand(n+1)
    alfa = 0.2
    itmax = 100
    emax = 0.01
    for t in range(itmax):
        errores = []
        for p in range(npe):
            x, d = train_data[p]
            net = dot(w, x)
            y = escalon(net)
            error = d - y
            errores.append(error)
            w = w + alfa * error * x
        print("errores de Train: ",errores)
        sw=0
        for p in range(npe):
            if fabs(errores[p])>emax:
                sw=1
                print("ERROR EN TRAIN")
    if sw==0:
        # realiza el TEST si ha aprendido en TRAIN
        # datos para test tomados dos de los 8 aprendizajes:
        test_data = [
            (array([15,250,1]), 1),
            (array([20,30,1]), 0) ]
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

#Haciendo la consulta mas segura para que nos de un valor certero
def consulta(w):
    query_data = [
        (array([9,2,1])),
        (array([11,240,1])) ]
    A= 1
    B= int(input("B<0 o -1> = "))
    escalon = lambda x: B if x < 0 else A
    print("resultado de la consulta")
    print("0 = es MELON")
    print("1 = es NARANJA")
    for xc in query_data:
        net = dot(xc, w)
        print("{} -> {}".format(xc[:2], escalon(net)))