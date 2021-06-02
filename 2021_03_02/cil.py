from math import pi
def cilindro():
    r=float(input('radio= '))
    h=float(input('altura= '))
    area=2*pi*r*(r+h)
    vol=pi*r*r*h
    print('area= ',area, 'volumen= ',vol )
