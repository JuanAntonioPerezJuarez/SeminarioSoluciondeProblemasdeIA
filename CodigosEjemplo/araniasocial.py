import numpy as np

# ======================================================
# Función objetivo (puedes cambiarla)
# Ejemplo: función de esfera f(x) = sum(x^2)
# ======================================================
def objective_function(x):
    return np.sum(x**2)


# ======================================================
# Parámetros del algoritmo
# ======================================================
def SSO(obj_func, dim=30, n_spiders=50, max_iter=100, lb=-100, ub=100):
    """
    Implementación del algoritmo Social Spider Optimization (SSO)
    :param obj_func: función objetivo a minimizar
    :param dim: número de dimensiones
    :param n_spiders: tamaño de la población
    :param max_iter: iteraciones máximas
    :param lb: límite inferior de búsqueda
    :param ub: límite superior de búsqueda
    :return: mejor solución y su valor objetivo
    """

    # Inicializar población (posiciones)
    pop = np.random.uniform(lb, ub, (n_spiders, dim))
    fitness = np.array([obj_func(ind) for ind in pop])

    # Asignación de sexos (70% hembras, 30% machos)
    n_females = int(0.7 * n_spiders)
    n_males = n_spiders - n_females

    # Ordenar población por fitness
    idx = np.argsort(fitness)
    pop = pop[idx]
    fitness = fitness[idx]

    females = pop[:n_females]
    males = pop[n_females:]
    f_fitness = fitness[:n_females]
    m_fitness = fitness[n_females:]

    # Parámetros de vibración
    def vibration(source_fit, target_fit, dist):
        return np.exp(-dist**2) * (source_fit / (target_fit + 1e-10))

    # Mejor global
    gbest = pop[0].copy()
    gbest_value = fitness[0]

    for t in range(max_iter):
        # =======================
        # Movimiento de hembras
        # =======================
        mean_fitness = np.mean(f_fitness)
        prob = np.random.rand(n_females)
        new_females = np.copy(females)

        for i in range(n_females):
            if prob[i] < 0.5:
                # Atracción hacia mejor o peor hembra
                idx_better = np.argmin(f_fitness)
                idx_worse = np.argmax(f_fitness)
                best_female = females[idx_better]
                worst_female = females[idx_worse]

                dist_better = np.linalg.norm(females[i] - best_female)
                dist_worse = np.linalg.norm(females[i] - worst_female)

                vib_better = vibration(f_fitness[idx_better], f_fitness[i], dist_better)
                vib_worse = vibration(f_fitness[idx_worse], f_fitness[i], dist_worse)

                alpha = np.random.rand()
                new_females[i] += alpha * (vib_better * (best_female - females[i])
                                           - vib_worse * (worst_female - females[i]))
            else:
                # Movimiento aleatorio
                new_females[i] += np.random.uniform(-1, 1, dim)

        # =======================
        # Movimiento de machos
        # =======================
        mean_male_fitness = np.mean(m_fitness)
        dominant_males = males[m_fitness < mean_male_fitness]
        submissive_males = males[m_fitness >= mean_male_fitness]
        new_males = np.copy(males)

        # Dominantes: atraídos por hembras
        for i in range(len(dominant_males)):
            female_idx = np.random.randint(0, n_females)
            dist = np.linalg.norm(dominant_males[i] - females[female_idx])
            vib = vibration(f_fitness[female_idx], obj_func(dominant_males[i]), dist)
            new_males[i] += np.random.rand() * vib * (females[female_idx] - dominant_males[i])

        # Sumisos: atraídos por mejores machos
        if len(submissive_males) > 0 and len(dominant_males) > 0:
            best_dominant = dominant_males[np.argmin([obj_func(x) for x in dominant_males])]
            for i in range(len(submissive_males)):
                dist = np.linalg.norm(submissive_males[i] - best_dominant)
                vib = vibration(obj_func(best_dominant), obj_func(submissive_males[i]), dist)
                submissive_males[i] += np.random.rand() * vib * (best_dominant - submissive_males[i])

            new_males = np.vstack((dominant_males, submissive_males))

        # =======================
        # Reemplazar machos y hembras
        # =======================
        females = np.clip(new_females, lb, ub)
        males = np.clip(new_males, lb, ub)

        # Recalcular fitness
        pop = np.vstack((females, males))
        fitness = np.array([obj_func(ind) for ind in pop])

        # Ordenar
        idx = np.argsort(fitness)
        pop = pop[idx]
        fitness = fitness[idx]

        # Actualizar mejor global
        if fitness[0] < gbest_value:
            gbest = pop[0].copy()
            gbest_value = fitness[0]

        # Actualizar roles
        females = pop[:n_females]
        males = pop[n_females:]
        f_fitness = fitness[:n_females]
        m_fitness = fitness[n_females:]

    return gbest, gbest_value


# ======================================================
# Ejemplo de uso
# ======================================================
if __name__ == "__main__":
    best_sol, best_val = SSO(objective_function, dim=30, n_spiders=50, max_iter=200, lb=-5, ub=5)
    print(" Mejor solución encontrada:", best_sol)
    print("Valor objetivo:", best_val)