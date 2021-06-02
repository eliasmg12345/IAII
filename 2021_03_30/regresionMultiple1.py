import numpy as np
import matplotlib.pyplot as plt
x=np.array([0,0,0,1,1,1.5,2,2,2,3,3,5,5,5])
y=np.array([3,4,5,3,5,4,2,3,5,3,4,1,2,3])
#calcular ajustes para diferentes grados
sols={}
for grado in range(1,6):
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
