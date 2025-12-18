import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def peaks_function(x, y):
    """
    Función peaks de MATLAB adaptada a Python
    Esta función tiene múltiples máximos y mínimos locales
    """
    z = 3 * (1 - x)**2 * np.exp(-(x**2) - (y + 1)**2) \
        - 10 * (x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
        - 1/3 * np.exp(-(x + 1)**2 - y**2)
    return z

def peaks_gradient(x, y):
    """
    Gradiente analítico de la función peaks
    Retorna las derivadas parciales respecto a x e y
    """
    # Términos de la función original
    term1 = 3 * (1 - x)**2 * np.exp(-(x**2) - (y + 1)**2)
    term2 = -10 * (x/5 - x**3 - y**5) * np.exp(-x**2 - y**2)
    term3 = -1/3 * np.exp(-(x + 1)**2 - y**2)
    
    # Derivada parcial respecto a x
    dz_dx = (3 * (-2 * (1 - x) * np.exp(-(x**2) - (y + 1)**2) + 
                  (1 - x)**2 * (-2 * x) * np.exp(-(x**2) - (y + 1)**2)) +
             -10 * ((1/5 - 3*x**2) * np.exp(-x**2 - y**2) + 
                    (x/5 - x**3 - y**5) * (-2*x) * np.exp(-x**2 - y**2)) +
             -1/3 * (-2 * (x + 1)) * np.exp(-(x + 1)**2 - y**2))
    
    # Derivada parcial respecto a y
    dz_dy = (3 * (1 - x)**2 * (-2 * (y + 1)) * np.exp(-(x**2) - (y + 1)**2) +
             -10 * ((-5 * y**4) * np.exp(-x**2 - y**2) + 
                    (x/5 - x**3 - y**5) * (-2*y) * np.exp(-x**2 - y**2)) +
             -1/3 * (-2 * y) * np.exp(-(x + 1)**2 - y**2))
    
    return dz_dx, dz_dy

def gradient_ascent(start_x, start_y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    """
    Algoritmo de gradiente ascendente (para maximización)
    
    Parámetros:
    - start_x, start_y: punto inicial
    - learning_rate: tasa de aprendizaje
    - max_iterations: número máximo de iteraciones
    - tolerance: tolerancia para convergencia
    """
    x, y = start_x, start_y
    history_x = [x]
    history_y = [y]
    history_z = [peaks_function(x, y)]
    
    for i in range(max_iterations):
        # Calcular gradiente
        grad_x, grad_y = peaks_gradient(x, y)
        
        # Actualizar posición (ASCENDENTE: sumamos el gradiente)
        x_new = x + learning_rate * grad_x
        y_new = y + learning_rate * grad_y
        
        # Verificar convergencia
        if np.sqrt((x_new - x)**2 + (y_new - y)**2) < tolerance:
            print(f"Convergencia alcanzada en iteración {i}")
            break
        
        x, y = x_new, y_new
        history_x.append(x)
        history_y.append(y)
        history_z.append(peaks_function(x, y))
    
    return x, y, peaks_function(x, y), history_x, history_y, history_z

def plot_optimization_results(history_x, history_y, history_z, final_x, final_y, final_z):
    """
    Grafica los resultados de la optimización
    """
    # Crear malla para la superficie
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = peaks_function(X, Y)
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Subplot 1: Vista 3D de la superficie con trayectoria
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
    ax1.plot(history_x, history_y, history_z, 'ro-', linewidth=2, markersize=4, label='Trayectoria')
    ax1.scatter([final_x], [final_y], [final_z], color='red', s=100, label=f'Máximo: ({final_x:.3f}, {final_y:.3f}, {final_z:.3f})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Superficie 3D con Trayectoria de Optimización')
    ax1.legend()
    
    # Subplot 2: Vista de contorno con trayectoria
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X, Y, Z, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(history_x, history_y, 'ro-', linewidth=2, markersize=4, label='Trayectoria')
    ax2.scatter([final_x], [final_y], color='red', s=100, label=f'Máximo: ({final_x:.3f}, {final_y:.3f})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Curvas de Nivel con Trayectoria')
    ax2.legend()
    ax2.grid(True)
    
    # Subplot 3: Convergencia del valor de la función
    ax3 = fig.add_subplot(133)
    ax3.plot(history_z, 'b-', linewidth=2, marker='o', markersize=4)
    ax3.set_xlabel('Iteración')
    ax3.set_ylabel('Valor de la Función')
    ax3.set_title('Convergencia del Algoritmo')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

# Ejecutar el algoritmo con diferentes puntos iniciales
def run_multiple_optimizations():
    """
    Ejecuta el algoritmo desde múltiples puntos iniciales
    """
    start_points = [(-2, -1), (1, 1), (-1, 2), (2, -2), (0, 0)]
    results = []
    
    print("=== ALGORITMO DE GRADIENTE ASCENDENTE PARA MAXIMIZACIÓN ===")
    print("Función objetivo: Peaks function")
    print("\nResultados desde diferentes puntos iniciales:")
    print("-" * 60)
    
    for i, (start_x, start_y) in enumerate(start_points):
        print(f"\nPunto inicial {i+1}: ({start_x}, {start_y})")
        
        final_x, final_y, final_z, hist_x, hist_y, hist_z = gradient_ascent(
            start_x, start_y, learning_rate=0.01, max_iterations=1000
        )
        
        results.append((final_x, final_y, final_z, hist_x, hist_y, hist_z))
        
        print(f"Máximo encontrado: ({final_x:.6f}, {final_y:.6f})")
        print(f"Valor máximo: {final_z:.6f}")
        print(f"Iteraciones: {len(hist_x)}")
    
    # Encontrar el mejor resultado (máximo global)
    best_idx = np.argmax([result[2] for result in results])
    best_result = results[best_idx]
    
    print(f"\n{'='*60}")
    print("MEJOR RESULTADO ENCONTRADO:")
    print(f"Punto: ({best_result[0]:.6f}, {best_result[1]:.6f})")
    print(f"Valor máximo: {best_result[2]:.6f}")
    print(f"{'='*60}")
    
    # Graficar el mejor resultado
    plot_optimization_results(best_result[3], best_result[4], best_result[5], 
                            best_result[0], best_result[1], best_result[2])
    
    return results

# Ejecutar el programa
if __name__ == "__main__":
    results = run_multiple_optimizations()
