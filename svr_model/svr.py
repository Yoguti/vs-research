import matplotlib.pyplot as plt
import numpy as np
import struct
from sklearn.svm import SVR

#Dataset aleatório pra teste
from sklearn.datasets import load_diabetes

# Carregar o dataset
data = load_diabetes()
#Matriz de entrada
X = data.data
#Vetor alvo
y = data.target

# Treinamento do SVR
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X, y)

# Extração dos parâmetros internos

#Pontos de margem
support_vectors = model.support_vectors_
#Coeficientes de Lagrange
dual_coef = model.dual_coef_
#Bias
intercept = model.intercept_

# Salvando parâmetros em binário
with open('svr_params.bin', 'wb') as f:
    f.write(struct.pack('I', len(support_vectors)))
    f.write(struct.pack('I', len(support_vectors[0])))  # dimensão de cada vetor

    # Vetores de suporte
    for vec in support_vectors:
        for val in vec:
            f.write(struct.pack('f', val))

    # Coeficientes (dual_coef)
    for coef in dual_coef[0]:
        f.write(struct.pack('f', coef))

    # Bias
    f.write(struct.pack('f', intercept[0]))

