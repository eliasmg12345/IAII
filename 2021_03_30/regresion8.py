def regresion8():
    import numpy as np
    import matplotlib.pyplot as plt
    #creacion de conjunto de datos para entrenamiento
    X=np.linspace(-2,2,101)
    Y=3+2*X+np.random.randn(*X.shape)*0.33
    #definicion de os ajustes y parametros iniciales
    num_steps=100
    alpha=0.10
    emax=1e-8
    b0=1
    b1=1
    #proceso iterativo
    for step in range(0,num_steps):
        suma_b0=0
        suma_b1=0
        M=float(len(X))
        for i in range(0,len(X)):
            suma_b0+=(b0+b1*X[i]-Y[i])
            suma_b1+= ((b0 + b1 * X[i] - Y[i]))*X[i]
        b0_grad=(2/M)*suma_b0
        b1_grad=(2/M)*suma_b1
        b0=b0-(alpha*b0_grad)
        b1 = b1 - (alpha * b1_grad)
        if max(abs(alpha*b0_grad),abs(alpha*b1_grad))<emax:
            break
            #impresion de los resultados
    print("b0=",b0,"b1=",b1,"en pasos", step)
    #grafio de datos y resultados
    f=np.poly1d([b1,b0])#polinomio de grado1 (recta)
    print("recta de regresio: ",f)
    yp=f(X)
    #grafica  de los datos(puntos)
    plt.plot(X,Y,linestyle='',marker='o',markersize=5)
    #grafica de la recta de regresion
    plt.plot(X,yp)
    plt.show()
