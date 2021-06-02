from sklearn import datasets, neighbors, linear_model
X_digitos,y_digitos=datasets.load_digits(return_X_y=True)
X_digitos=X_digitos/X_digitos.max()
n_muestras=len(X_digitos)
X_entrenar=X_digitos[:int(.9*n_muestras)]
y_entrenar=y_digitos[:int(.9*n_muestras)]
print('datos de entrenar: ')
print('x de entrenar : ',X_entrenar)
print('y de entrenar: ',y_entrenar)
