import numpy as np
import matplotlib.pyplot as plt
x=np.array([0.75,1,1.25,1.5,1.65,1.75,2,2.25,2.5,2.75,3,3.25,3.5,4,4.25,4.5,4.75,5,5.25])
y=np.array([10,15,17,20,22,24,30,35,42,48,56,62,68,73,78,80,83,85,89])
#calcular ajustes para diferentes grados
sols={}
for grado in range(1,3):
    z=np.polyfit(x,y,grado,full=True)
    sols[grado]=z
#sibujar datos
plt.plot(x,y,'o')
#Dibujar  curvas de ajuste
xp=np.linspace(0,5.2,100)
for grado, sol in sols.items():
    coefs,error, *_=sol
    p=np.poly1d(coefs)
    plt.plot(xp,p(xp),"-",label="Gr:%s.Error%.3f"%(grado,error))
plt.legend()
plt.show()
