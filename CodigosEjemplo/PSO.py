import numpy as np

# Función objetivo (función de esfera)
def funcion_objetivo(x):
    return np.sum(x**2)

# Parámetros del algoritmo
num_particulas = 30
dimensiones = 2
limites = (-10, 10)
max_iter = 100

w = 0.5      # inercia
c1 = 1.5     # coeficiente cognitivo
c2 = 1.5     # coeficiente social

# Inicialización aleatoria de partículas
posiciones = np.random.uniform(limites[0], limites[1], (num_particulas, dimensiones))
velocidades = np.random.uniform(-1, 1, (num_particulas, dimensiones))
mejor_pos_personal = np.copy(posiciones)
mejor_val_personal = np.array([funcion_objetivo(p) for p in posiciones])

# Mejor posición global
mejor_idx = np.argmin(mejor_val_personal)
mejor_pos_global = mejor_pos_personal[mejor_idx]

# Bucle principal de PSO
for iteracion in range(max_iter):
    for i in range(num_particulas):
        r1, r2 = np.random.rand(), np.random.rand()

        # Actualizar velocidad
        velocidades[i] = (
            w * velocidades[i]
            + c1 * r1 * (mejor_pos_personal[i] - posiciones[i])
            + c2 * r2 * (mejor_pos_global - posiciones[i])
        )

        # Actualizar posición
        posiciones[i] += velocidades[i]

        # Limitar dentro del rango
        posiciones[i] = np.clip(posiciones[i], limites[0], limites[1])

        # Evaluar
        valor_actual = funcion_objetivo(posiciones[i])

        # Actualizar mejor personal
        if valor_actual < mejor_val_personal[i]:
            mejor_val_personal[i] = valor_actual
            mejor_pos_personal[i] = posiciones[i]

    # Actualizar mejor global
    mejor_idx = np.argmin(mejor_val_personal)
    mejor_pos_global = mejor_pos_personal[mejor_idx]

    print(f"Iteración {iteracion+1}: Mejor valor = {mejor_val_personal[mejor_idx]:.4f}")

# Resultado final
print("\nMejor solución encontrada:")
print("Posición:", mejor_pos_global)
print("Valor:", funcion_objetivo(mejor_pos_global))
