import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Definición de la función de activación (Sigmoide) y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Datos de entrenamiento: XOR de 3 entradas
# Regla XOR 3 inputs: Salida 1 si el número de 1s es impar.
# [A, B, C]
X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Salidas esperadas (Targets)
y = np.array([
    [0], # 0 unos (par) -> 0
    [1], # 1 uno (impar) -> 1
    [1], # 1 uno (impar) -> 1
    [0], # 2 unos (par) -> 0
    [1], # 1 uno (impar) -> 1
    [0], # 2 unos (par) -> 0
    [0], # 2 unos (par) -> 0
    [1]  # 3 unos (impar) -> 1
])

# 3. Inicialización de la Red Neuronal
np.random.seed(42) # Para reproducibilidad

# Arquitectura: 3 Entradas -> 4 Neuronas Ocultas -> 1 Salida
input_layer_neurons = 3
hidden_layer_neurons = 4
output_neurons = 1

# Pesos y sesgos (inicializados aleatoriamente)
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))
wout = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Hiperparámetros
lr = 0.1       # Tasa de aprendizaje (Learning Rate)
epochs = 20000 # Número de iteraciones

# 4. Entrenamiento (Backpropagation)
errores = []

print("Entrenando la red neuronal...")

for i in range(epochs):
    # --- Forward Propagation (Paso hacia adelante) ---
    hidden_layer_input1 = np.dot(X, wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hidden_layer_activations = sigmoid(hidden_layer_input)
    
    output_layer_input1 = np.dot(hidden_layer_activations, wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)
    
    # --- Backpropagation (Paso hacia atrás) ---
    # Calcular el error
    E = y - output
    errores.append(np.mean(np.abs(E))) # Guardar error para graficar
    
    # Calcular gradientes
    slope_output_layer = sigmoid_derivative(output)
    slope_hidden_layer = sigmoid_derivative(hidden_layer_activations)
    
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    
    # Actualizar pesos y sesgos
    wout += hidden_layer_activations.T.dot(d_output) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hiddenlayer) * lr
    bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr

print(f"Entrenamiento finalizado. Error final: {errores[-1]:.5f}")

# 5. Evaluación y Visualización
output_binario = np.round(output) # Redondear a 0 o 1

# Crear figura
fig = plt.figure(figsize=(14, 6))

# Subplot 1: Curva de error
ax1 = fig.add_subplot(121)
ax1.plot(errores)
ax1.set_title('Convergencia del Error')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Error Medio')
ax1.grid(True)

# Subplot 2: Visualización 3D de la clasificación
ax2 = fig.add_subplot(122, projection='3d')

# Separar puntos por clase predicha para colorearlos
x_0 = X[output_binario.flatten() == 0]
x_1 = X[output_binario.flatten() == 1]

# Graficar puntos clase 0 (Rojos)
ax2.scatter(x_0[:,0], x_0[:,1], x_0[:,2], c='red', s=100, label='Clase 0 (Par)', marker='o')
# Graficar puntos clase 1 (Azules)
ax2.scatter(x_1[:,0], x_1[:,1], x_1[:,2], c='blue', s=100, label='Clase 1 (Impar)', marker='^')

ax2.set_xlabel('Entrada A')
ax2.set_ylabel('Entrada B')
ax2.set_zlabel('Entrada C')
ax2.set_title('Clasificación XOR 3D (Predicción)')
ax2.legend()

plt.tight_layout()
plt.show()

# Mostrar predicciones numéricas
print("\n--- Resultados Finales ---")
print("Entrada    | Real | Predicción")
print("------------------------------")
for i in range(len(X)):
    print(f"{X[i]}  |  {y[i][0]}   |  {output[i][0]:.4f} -> {int(output_binario[i][0])}")

