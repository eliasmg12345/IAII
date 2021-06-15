#ejemplo perceptron multicapa para XOR
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, ylim, xlim
#funcion sigmoide y derivada
def sigmoide (x):
    return 1 / ( 1 + np.exp(-x))
def d_sigmoide(x):
    return 1 / ( 2 + np.exp(-x)+np.exp(x))
def aprendizaje():
    np.set_printoptions(precision=6)
    x = np.array([[0, 0], [0, 1],[1, 0], [1, 1]])
    d = np.array([[0], [1], [1], [0]])
    alfa = 1; itmax = 3000; muestras = 9;
    ne= len(x[0]) # Número de neuronas capa de entrada
    ns= len(d[0]) # Número de neuronas capa de salida
    # Número de neuronas en la capa oculta
    no = int(input("No. neuronas en la capa oculta= "))
    error_plot = []
    # Inicializar aleatoriamente pesos de las conexiones:
    # W0 de neuronas capa oculta
    bias1= np.random.rand()
    # W0 de neuronas capa salida
    bias2=np.random.rand()
    # w1: pesos capa entrada y oculta NoEntradas*n_oculta
    w1=np.zeros((ne,no))
    for k in range(ne):
        w1[k]=np.array(np.random.rand(no))
    # w2: pesos capa oculta y salida n_oculta*NoSalidas
    w2=np.zeros((no,ns))
    for k in range(no):
        w2[k]=np.array(np.random.rand(ns))
    #iteraciones
    for t in range(itmax):
        #propagacion hacia adelante (obtener la salida de la red)
        Net_h = np.dot(x, w1) + bias1
        Y_h = sigmoide(Net_h) # salida de la capa oculta
        Net_o = np.dot(Y_h, w2) + bias2
        Y_o = sigmoide(Net_o) # salida de la red
        #errorCuadraticoMedio=1/2*(salidaObtenida-salidaDeseada)^2
        error_salida_o=0.5*(np.power((Y_o - d),2))
        promedio_error = (np.mean(error_salida_o) )
        error_plot.append(promedio_error)
        #propagacion hacia atras
        # capa de salida
        d_error_salida_o = Y_o - d
        d_salida_derivada = d_sigmoide(Y_o)
        ## Errores de la capa de salida
        d_capa_salida = np.dot(Y_h.T, (d_error_salida_o * d_salida_derivada))
        # capa oculta
        d_salida_h=np.dot(d_error_salida_o * d_salida_derivada, w2.T)
        d_entrada_h = d_sigmoide(Y_h)
        ## Errores de la capa oculta
        d_capa_oculta = np.dot(x.T, d_salida_h * d_entrada_h)
        # ACTUALIZACIÓN PESOS CAPAS OCULTA Y SALIDA
        ## wi = wi - alfa * delta*entradai
        w1 = w1 - alfa * d_capa_oculta
        w2 = w2 - alfa * d_capa_salida
        if (t% muestras) == 0:
            print("Iteracion (%s)-->Error: %s--> salida %s" % (t+1, promedio_error, Y_o))
    #Gráfico (solo para dos entradas y una salida)
    if ne<=2 and ns==1:
        ylim([0,0.3])
        xlim([0,itmax])
        plt.xlabel("Iteraciones")
        plt.ylabel("Error")
        plt.plot(error_plot, color='r')
        plt.title("Perceptron Multicapa"); plt.grid(True)
        plt.show()