import pylab as pl
import numpy as np
def grafico1():
    pl.figure(figsize=(8,6),dpi=8)
    pl.subplot(1,1,1)
    X=np.linspace(-np.pi,np.pi,256,endpoint=True)
    C=np.cos(X)
    S=np.sin(X)
    pl.plot(X,C,color="blue",linewidth=1.0,linestyle="-")
    pl.plot(X,S,color="green",linewidth=1.0,linestyle="-")
    pl.xlim(-4.0,4.0)
    pl.xticks(np.linspace(-4,4,9,endpoint=True))
    pl.ylim(-1.0,1.0)
    pl.yticks(np.linspace(-1,1,5,endpoint=True))
    pl.show()

grafico1()