import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------
# Definición de funciones
# ---------------------

# Función peaks
def peaks(x, y):
    return 3*(1 - x)**2 * np.exp(-(x**2) - (y + 1)**2) - \
           10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) - \
           1/3*np.exp(-(x + 1)**2 - y**2)

# Función objetivo que toma un vector [x, y]
def objective_function(position):
    x, y = position
    return peaks(x, y)

# Gradiente analítico de la función peaks
def peaks_gradient(x, y):
    # Término 1: 3*(1-x)² * exp(-(x²+(y+1)²))
    exp1 = np.exp(-(x**2) - (y + 1)**2)
    term1 = 3 * (1 - x)**2 * exp1
    
    # Derivadas del término 1
    dterm1_dx = 3 * (-2*(1-x)*exp1 + (1-x)**2*(-2*x)*exp1)
    dterm1_dy = 3 * (1-x)**2 * (-2*(y+1)) * exp1
    
    # Término 2: -10*(x/5 - x³ - y⁵) * exp(-x² - y²)
    exp2 = np.exp(-x**2 - y**2)
    inner2 = x/5 - x**3 - y**5
    term2 = -10 * inner2 * exp2
    
    # Derivadas del término 2
    dterm2_dx = -10 * ((1/5 - 3*x**2)*exp2 + inner2*(-2*x)*exp2)
    dterm2_dy = -10 * ((-5*y**4)*exp2 + inner2*(-2*y)*exp2)
    
    # Término 3: -1/3 * exp(-(x+1)² - y²)
    exp3 = np.exp(-(x + 1)**2 - y**2)
    term3 = -1/3 * exp3
    
    # Derivadas del término 3
    dterm3_dx = -1/3 * (-2*(x+1)) * exp3
    dterm3_dy = -1/3 * (-2*y) * exp3
    
    # Gradiente total
    grad_x = dterm1_dx + dterm2_dx + dterm3_dx
    grad_y = dterm1_dy + dterm2_dy + dterm3_dy
    
    return np.array([grad_x, grad_y])

# Función de gradiente que toma un vector posición
def gradient_function(position):
    x, y = position
    return peaks_gradient(x, y)

# Límites del espacio de búsqueda
lower_bound = np.array([-3, -3])
upper_bound = np.array([3, 3])

# Función para mantener dentro de los límites
def clip_to_bounds(position):

    return np.clip(position, lower_bound, upper_bound)

# ---------------------
# ALGORITMO DE GRADIENTE ASCENDENTE
# ---------------------

def gradient_ascent(initial_solution, learning_rate=0.01, 
                   max_iterations=1000, tolerance=1e-6,
                   adaptive=True, decay_rate=0.999, verbose=True):
 
    current_solution = initial_solution.copy()
    current_energy = objective_function(current_solution)
    
    best_solution = current_solution.copy()
    best_energy = current_energy
    
    # Historial para análisis
    path = [current_solution.copy()]
    energy_history = [current_energy]
    lr_history = [learning_rate]
    
    current_lr = learning_rate
    
    if verbose:
        print("Iniciando Gradiente Ascendente para maximización")
        print(f"Punto inicial: ({current_solution[0]:.4f}, {current_solution[1]:.4f})")
        print(f"Valor inicial: {current_energy:.6f}")
        print(f"Tasa de aprendizaje: {learning_rate}")
        print("-" * 60)
    
    for iteration in range(max_iterations):
        # Calcular gradiente
        gradient = gradient_function(current_solution)
        gradient_magnitude = np.linalg.norm(gradient)
        
        # Verificar convergencia por gradiente pequeño
        if gradient_magnitude < tolerance:
            if verbose:
                print(f"Convergencia por gradiente pequeño en iteración {iteration}")
            break
        
        # Actualizar posición (ascendente: sumar gradiente)
        new_solution = current_solution + current_lr * gradient
        
        # Mantener dentro de los límites
        new_solution = clip_to_bounds(new_solution)
        new_energy = objective_function(new_solution)
        
        # Tasa de aprendizaje adaptativa
        if adaptive:
            if new_energy > current_energy:
                # Mejora: aumentar ligeramente la tasa
                current_lr = min(current_lr * 1.01, learning_rate * 2)
            else:
                # Empeora: reducir la tasa
                current_lr *= 0.8
                if current_lr < 1e-10:
                    if verbose:
                        print(f"Tasa de aprendizaje muy pequeña en iteración {iteration}")
                    break
        else:
            # Decaimiento simple
            current_lr *= decay_rate
        
        # Verificar convergencia por movimiento pequeño
        movement = np.linalg.norm(new_solution - current_solution)
        if movement < tolerance:
            if verbose:
                print(f"Convergencia por movimiento pequeño en iteración {iteration}")
            break
        
        # Actualizar solución actual
        current_solution = new_solution
        current_energy = new_energy
        
        # Actualizar mejor solución (maximización)
        if current_energy > best_energy:
            best_solution = current_solution.copy()
            best_energy = current_energy
        
        # Guardar historial
        path.append(current_solution.copy())
        energy_history.append(current_energy)
        lr_history.append(current_lr)
        
        # Mostrar progreso
        if verbose and iteration % 100 == 0:
            print(f"Iter {iteration:4d}: ({current_solution[0]:8.4f}, {current_solution[1]:8.4f}) "
                  f"= {current_energy:10.6f}, lr = {current_lr:.2e}")
    
    if verbose:
        print("-" * 60)
        print("RESULTADO FINAL:")
        print(f"Mejor solución: ({best_solution[0]:.6f}, {best_solution[1]:.6f})")
        print(f"Valor máximo: {best_energy:.6f}")
        print(f"Iteraciones: {len(path)}")
    
    return best_solution, best_energy, np.array(path), energy_history, lr_history

# ---------------------
# Función para múltiples ejecuciones
# ---------------------

def multiple_gradient_ascent_runs(num_runs=5, verbose=True):
    if verbose:
        print(f"Ejecutando {num_runs} corridas desde puntos aleatorios")
        print("=" * 70)
    
    all_results = []
    
    for run in range(num_runs):
        if verbose:
            print(f"\nCORRIDA {run + 1}/{num_runs}")
        
        # Punto inicial aleatorio
        initial_solution = np.random.uniform(lower_bound, upper_bound)
        
        # Ejecutar algoritmo
        best_sol, best_val, path, energy_hist, lr_hist = gradient_ascent(
            initial_solution, 
            learning_rate=0.05,
            max_iterations=500,
            adaptive=True,
            verbose=verbose
        )
        
        all_results.append({
            'solution': best_sol,
            'value': best_val,
            'path': path,
            'energy_history': energy_hist,
            'lr_history': lr_hist,
            'initial': initial_solution
        })
    
    # Encontrar el mejor resultado global
    best_global_idx = np.argmax([r['value'] for r in all_results])
    best_global = all_results[best_global_idx]
    
    if verbose:
        print(f"\nMEJOR RESULTADO GLOBAL (Corrida {best_global_idx + 1}):")
        print(f"Punto inicial: ({best_global['initial'][0]:.4f}, {best_global['initial'][1]:.4f})")
        print(f"Máximo encontrado: ({best_global['solution'][0]:.6f}, {best_global['solution'][1]:.6f})")
        print(f"Valor máximo: {best_global['value']:.6f}")
    
    return all_results, best_global

# ---------------------
# Visualización
# ---------------------

def plot_optimization_results(result, show_3d=True):
    path = result['path']
    energy_history = result['energy_history']
    lr_history = result['lr_history']
    best_solution = result['solution']
    best_energy = result['value']
    initial_solution = result['initial']
    
    # Crear malla para el gráfico
    x = np.linspace(lower_bound[0], upper_bound[0], 200)
    y = np.linspace(lower_bound[1], upper_bound[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = peaks(X, Y)
    
    # Convertir trayectoria a arrays
    px, py = path[:, 0], path[:, 1]
    pz = [peaks(p[0], p[1]) for p in path]
    
    if show_3d:
        # Crear figura con múltiples subplots
        fig = plt.figure(figsize=(20, 6))
        
        # Subplot 1: Vista 3D
        ax1 = fig.add_subplot(141, projection='3d')
        surface = ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='plasma')
        ax1.plot(px, py, pz, 'ro-', linewidth=3, markersize=4, label='Trayectoria')
        ax1.scatter([initial_solution[0]], [initial_solution[1]], [peaks(initial_solution[0], initial_solution[1])], 
                   color='green', s=100, label='Inicio', marker='s')
        ax1.scatter([best_solution[0]], [best_solution[1]], [best_energy], 
                   color='red', s=150, label=f'Máximo: {best_energy:.4f}', marker='*')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Vista 3D - Función Peaks')
        ax1.legend()
        
        # Subplot 2: Contornos
        ax2 = fig.add_subplot(142)
    else:
        fig = plt.figure(figsize=(15, 5))
        ax2 = fig.add_subplot(131)
    
    # Contornos de la función
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='plasma')
    plt.colorbar(contour, ax=ax2, label='f(x, y)')
    
    # Ruta seguida por el algoritmo
    ax2.plot(px, py, color='white', linestyle='-', linewidth=3, label='Trayectoria', alpha=0.9)
    
    # Puntos importantes
    ax2.plot(initial_solution[0], initial_solution[1], 'o', color='green', 
             markersize=10, label='Inicio', markeredgecolor='black')
    ax2.plot(best_solution[0], best_solution[1], '*', color='red', 
             markersize=15, label=f'Máximo: {best_energy:.4f}', markeredgecolor='black')
    
    ax2.set_title("Gradiente Ascendente - Función Peaks (Maximización)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(lower_bound[0], upper_bound[0])
    ax2.set_ylim(lower_bound[1], upper_bound[1])
    
    # Subplot 3: Convergencia de la función
    ax3 = fig.add_subplot(143 if show_3d else 132)
    iterations = range(len(energy_history))
    ax3.plot(iterations, energy_history, 'b-', linewidth=2, marker='o', markersize=3)
    ax3.axhline(y=best_energy, color='red', linestyle='--', alpha=0.7, 
               label=f'Máximo: {best_energy:.6f}')
    ax3.set_xlabel('Iteración')
    ax3.set_ylabel('Valor de la Función')
    ax3.set_title('Convergencia del Algoritmo')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Subplot 4: Evolución de la tasa de aprendizaje
    ax4 = fig.add_subplot(144 if show_3d else 133)
    ax4.plot(iterations, lr_history, 'g-', linewidth=2, marker='s', markersize=2)
    ax4.set_xlabel('Iteración')
    ax4.set_ylabel('Tasa de Aprendizaje')
    ax4.set_title('Evolución de la Tasa de Aprendizaje')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# ---------------------
# EJECUCIÓN PRINCIPAL
# ---------------------

if __name__ == "__main__":
    print("ALGORITMO DE GRADIENTE ASCENDENTE PARA MAXIMIZACIÓN")
    print("Función objetivo: PEAKS")
    print("=" * 70)
    
    # Ejecutar múltiples corridas
    all_results, best_global_result = multiple_gradient_ascent_runs(num_runs=6)
    
    # Visualizar el mejor resultado
    print("\nVisualizando mejor resultado...")
    plot_optimization_results(best_global_result, show_3d=True)
    
    # Estadísticas de todas las corridas
    print("\nESTADÍSTICAS DE TODAS LAS CORRIDAS:")
    print("-" * 50)
    values = [r['value'] for r in all_results]
    print(f"Mejor valor: {max(values):.6f}")
    print(f"Valor promedio: {np.mean(values):.6f}")
    print(f"Peor valor: {min(values):.6f}")
    print(f"Desviación estándar: {np.std(values):.6f}")
    
    # Comparar con diferentes configuraciones
    print("\nProbando diferentes configuraciones...")
    
    # Configuración conservadora
    print("\nConfiguración CONSERVADORA (lr=0.01, no adaptativo):")
    initial_solution = np.array([1.5, -0.5])
    result_conservative, _, path_cons, energy_cons, lr_cons = gradient_ascent(
        initial_solution, learning_rate=0.01, adaptive=False, max_iterations=1000
    )
    
    # Configuración agresiva
    print("\nConfiguración AGRESIVA (lr=0.1, adaptativo):")
    result_aggressive, _, path_aggr, energy_aggr, lr_aggr = gradient_ascent(
        initial_solution, learning_rate=0.1, adaptive=True, max_iterations=500
    )
    
    print("\nCOMPARACIÓN DE CONFIGURACIONES:")
    print(f"Conservadora: {result_conservative} = {peaks(result_conservative[0], result_conservative[1]):.6f}")
    print(f"Agresiva:     {result_aggressive} = {peaks(result_aggressive[0], result_aggressive[1]):.6f}")
    
    print("\nOptimización completada.")
    print("El algoritmo de gradiente ascendente encontró el máximo de la función peaks.")
