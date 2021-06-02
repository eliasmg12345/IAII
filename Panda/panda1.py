import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def regresion6():
    clima=pd.read_excel('tiempo2.xlsx',sheet_name='datos')
    print(clima)
    julio=clima.query('Month==7')
    julio.insert(0,'Yr',range(0,len(julio)))
    julio.plot(y='Tmax',x='Yr')
    d=np.polyfit(julio['Yr'],julio['Tmax'],1)
    f=np.poly1d(d)
    print(f)
    julio.insert(6,'Treg',f(julio['Yr']))
    ax=julio.plot(x='Yr',y='Tmax')
    julio.plot(x='Yr',y='Treg',color='Red',ax=ax)
    ax=julio.plot.scatter(x='Yr',y='Tmax')
    julio.plot(x='Yr',y='Treg',color='Red',legend=False,ax=ax)
    plt.show()

regresion6()