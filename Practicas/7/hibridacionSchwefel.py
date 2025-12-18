import numpy as np

# -----------------------------
# Función objetivo: Rastrigin
# -----------------------------
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# -----------------------------
# Función objetivo: Schwefel
# -----------------------------
def schwefel(x):
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# Gradiente de la función Rastrigin
def grad_rastrigin(x):
    A = 10
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

# Gradiente de la función Schwefel
def grad_schwefel(x):
    return np.sin(np.sqrt(np.abs(x))) + (x / np.abs(x)) * np.cos(np.sqrt(np.abs(x)))

# -----------------------------
# Etapa 1: Gradiente Descendente (optimización gruesa)
# -----------------------------
def gradient_descent(func, grad_func, x0, lr=0.01, max_iter=200):
    x = x0.copy()
    for i in range(max_iter):
        grad = grad_func(x)
        x -= lr * grad
    return x

# -----------------------------
# Etapa 2: PSO (optimización fina)
# -----------------------------
def pso(func, dim, bounds, num_particles=30, max_iter=100, init_pos=None):
    w = 0.9   # inercia ajustada
    c1 = 2.0  # coeficiente cognitivo ajustado
    c2 = 2.0  # coeficiente social ajustado

    # Inicializar posiciones
    if init_pos is not None:
        X = np.random.uniform(low=bounds[0], high=bounds[1], size=(num_particles, dim))
        X = init_pos + 0.1 * (X - init_pos)  # pequeña perturbación
    else:
        X = np.random.uniform(low=bounds[0], high=bounds[1], size=(num_particles, dim))
        
    V = np.zeros_like(X)
    pbest = X.copy()
    pbest_val = np.array([func(x) for x in X])
    gbest = X[np.argmin(pbest_val)]
    gbest_val = np.min(pbest_val)

    for t in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            X[i] += V[i]
            X[i] = np.clip(X[i], bounds[0], bounds[1])

            val = func(X[i])
            if val < pbest_val[i]:
                pbest[i], pbest_val[i] = X[i].copy(), val

        # Actualizar mejor global
        if np.min(pbest_val) < gbest_val:
            gbest = pbest[np.argmin(pbest_val)].copy()
            gbest_val = np.min(pbest_val)

    return gbest, gbest_val

# -----------------------------
# Ejecución del híbrido GD + PSO
# -----------------------------
if __name__ == "__main__":
    # np.random.seed(42)  # Esta línea fue eliminada

    dim = 2
    bounds = [-500, 500]

    # Paso 1: Gradiente Descendente (búsqueda gruesa)
    x0 = np.random.uniform(bounds[0], bounds[1], dim)
    x_gd = gradient_descent(schwefel, grad_schwefel, x0, lr=0.01, max_iter=500)  # Ajuste en max_iter
    print("Resultado de GD:", x_gd, "Valor:", schwefel(x_gd))

    # Paso 2: PSO (búsqueda fina)
    best_pso, best_val = pso(schwefel, dim, bounds, num_particles=50, max_iter=300, init_pos=x_gd)  # Ajustes en num_particles y max_iter
    print("Resultado final híbrido GD+PSO:", best_pso, "Valor:", best_val)
