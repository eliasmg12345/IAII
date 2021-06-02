def grafico7():
    import pylab as plt
    import numpy as np
    mes=range(1,13)
    prec=[98.8,128,130,100,100,110,94,120,116,103,101,95]
    plt.bar(mes,prec)
    plt.xticks(mes)
    plt.yticks(range(0,300,50))
    plt.grid(True,alpha=0.5,linestyle='-')
    plt.title('lluvia 2020')
    plt.ylabel('total lluvia(mm)')
    plt.xlabel('mes')
    plt.show()

grafico7()