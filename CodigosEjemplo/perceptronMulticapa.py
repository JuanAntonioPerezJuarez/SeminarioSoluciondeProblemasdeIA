import math
import random

# Datos de entrada y salida
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

# Generar pesos y sesgos aleatorios
def rand():
    return random.uniform(-1, 1)

# Inicialización de pesos y sesgos
w_hidden = [[rand(), rand()], [rand(), rand()]]
b_hidden = [rand(), rand()]
w_out = [rand(), rand()]
b_out = rand()

eta = 1.5 # Tasa de aprendizaje

# Entrenamiento
for epoch in range(5000):
    for i in range(len(x)):
        x1, x2 = x[i]
        target = y[i][0]

        # Capa Oculta
        z1 = w_hidden[0][0] * x1 + w_hidden[0][1] * x2 + b_hidden[0]
        h1 = sigmoid(z1)
        z2 = w_hidden[1][0] * x1 + w_hidden[1][1] * x2 + b_hidden[1]
        h2 = sigmoid(z2)

        # Capa de Salida
        z_out = w_out[0] * h1 + w_out[1] * h2 + b_out
        pred = sigmoid(z_out)
        error = pred - target

        # Retropropagación
        d_out = error * d_sigmoid(z_out)
        d_h1 = d_out * w_out[0] * d_sigmoid(z1)
        d_h2 = d_out * w_out[1] * d_sigmoid(z2)

        # Actualización de pesos y sesgos
        w_out[0] -= eta * d_out * h1
        w_out[1] -= eta * d_out * h2  # Corrección aquí
        b_out -= eta * d_out

        w_hidden[0][0] -= eta * d_h1 * x1
        w_hidden[0][1] -= eta * d_h1 * x2
        b_hidden[0] -= eta * d_h1

        w_hidden[1][0] -= eta * d_h2 * x1
        w_hidden[1][1] -= eta * d_h2 * x2
        b_hidden[1] -= eta * d_h2

# Resultados finales del MLP
print("\nResultados finales del MLP")
for i in range(len(x)):
    x1, x2 = x[i]
    z1 = w_hidden[0][0] * x1 + w_hidden[0][1] * x2 + b_hidden[0]
    h1 = sigmoid(z1)
    z2 = w_hidden[1][0] * x1 + w_hidden[1][1] * x2 + b_hidden[1]
    h2 = sigmoid(z2)

    z_out = w_out[0] * h1 + w_out[1] * h2 + b_out
    pred = sigmoid(z_out)
    print(f"x{i} --> {pred:.4f}")  # Corrección aquí


    #PowerWolf Blessed and possesed 