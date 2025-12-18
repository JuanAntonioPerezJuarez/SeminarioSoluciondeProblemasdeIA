import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def euler_function(x, y):
    """
    Función basada en la constante de Euler (e ≈ 2.718)
    Esta función combina exponenciales y trigonométricas para crear
    una superficie con múltiples máximos y mínimos
    """
    # Función que combina exponenciales con términos trigonométricos
    # Inspirada en la constante de Euler
    z = np.exp(-(x**2 + y**2)/4) * (np.cos(x) + np.sin(y)) + \
        0.5 * np.exp(-((x-2)**2 + (y-1)**2)/3) * np.cos(2*x + y) + \
        0.3 * np.exp(-((x+1)**2 + (y+2)**2)/2) * np.sin(x - y)
    return z

def euler_gradient(x, y):
    """
    Gradiente analítico de la función de Euler
    Calculamos las derivadas parciales respecto a x e y
    """
    # Término 1: exp(-(x²+y²)/4) * (cos(x) + sin(y))
    exp1 = np.exp(-(x**2 + y**2)/4)
    trig1 = np.cos(x) + np.sin(y)
    
    # Derivadas del término 1
    dterm1_dx = exp1 * (-x/2 * trig1 - np.sin(x))
    dterm1_dy = exp1 * (-y/2 * trig1 + np.cos(y))
    
    # Término 2: 0.5 * exp(-((x-2)²+(y-1)²)/3) * cos(2x+y)
    exp2 = 0.5 * np.exp(-((x-2)**2 + (y-1)**2)/3)
    cos2 = np.cos(2*x + y)
    
    # Derivadas del término 2
    dterm2_dx = exp2 * (-2*(x-2)/3 * cos2 - 2*np.sin(2*x + y))
    dterm2_dy = exp2 * (-2*(y-1)/3 * cos2 - np.sin(2*x + y))
    
    # Término 3: 0.3 * exp(-((x+1)²+(y+2)²)/2) * sin(x-y)
    exp3 = 0.3 * np.exp(-((x+1)**2 + (y+2)**2)/2)
    sin3 = np.sin(x - y)
    
    # Derivadas del término 3
    dterm3_dx = exp3 * (-(x+1) * sin3 + np.cos(x - y))
    dterm3_dy = exp3 * (-(y+2) * sin3 - np.cos(x - y))
    
    # Gradiente total
    dz_dx = dterm1_dx + dterm2_dx + dterm3_dx
    dz_dy = dterm1_dy + dterm2_dy + dterm3_dy
    
    return dz_dx, dz_dy

def gradient_ascent_euler(start_x, start_y, learning_rate=0.1, max_iterations=1000, tolerance=1e-6):
    """
    Algoritmo de gradiente ascendente para maximizar la función de Euler
    
    Parámetros:
    - start_x, start_y: punto inicial
    - learning_rate: tasa de aprendizaje
    - max_iterations: número máximo de iteraciones
    - tolerance: tolerancia para convergencia
    """
    x, y = start_x, start_y
    history_x = [x]
    history_y = [y]
    history_z = [euler_function(x, y)]
    
    print(f"Iniciando desde: ({x:.3f}, {y:.3f}), valor inicial: {history_z[0]:.6f}")
    
    for i in range(max_iterations):
        # Calcular gradiente
        grad_x, grad_y = euler_gradient(x, y)
        
        # Verificar si el gradiente es muy pequeño (posible máximo local)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        if gradient_magnitude < tolerance:
            print(f"Convergencia por gradiente pequeño en iteración {i}")
            break
        
        # Actualizar posición (ASCENDENTE: sumamos el gradiente)
        x_new = x + learning_rate * grad_x
        y_new = y + learning_rate * grad_y
        
        # Verificar convergencia por movimiento pequeño
        movement = np.sqrt((x_new - x)**2 + (y_new - y)**2)
        if movement < tolerance:
            print(f"Convergencia por movimiento pequeño en iteración {i}")
            break
        
        x, y = x_new, y_new
        current_z = euler_function(x, y)
        
        history_x.append(x)
        history_y.append(y)
        history_z.append(current_z)
        
        # Mostrar progreso cada 100 iteraciones
        if i % 100 == 0:
            print(f"Iteración {i}: ({x:.6f}, {y:.6f}), valor: {current_z:.6f}")
    
    final_z = euler_function(x, y)
    print(f"Resultado final: ({x:.6f}, {y:.6f}), valor máximo: {final_z:.6f}")
    
    return x, y, final_z, history_x, history_y, history_z

def adaptive_gradient_ascent(start_x, start_y, initial_lr=0.1, max_iterations=1000):
    """
    Versión con tasa de aprendizaje adaptativa
    Si el valor de la función disminuye, reducimos la tasa de aprendizaje
    """
    x, y = start_x, start_y
    learning_rate = initial_lr
    history_x = [x]
    history_y = [y]
    history_z = [euler_function(x, y)]
    history_lr = [learning_rate]
    
    print(f"🚀 Gradiente Ascendente Adaptativo")
    print(f"Punto inicial: ({x:.3f}, {y:.3f}), valor: {history_z[0]:.6f}")
    
    for i in range(max_iterations):
        current_z = euler_function(x, y)
        grad_x, grad_y = euler_gradient(x, y)
        
        # Proponer nuevo punto
        x_new = x + learning_rate * grad_x
        y_new = y + learning_rate * grad_y
        new_z = euler_function(x_new, y_new)
        
        # Si el nuevo valor es peor, reducir tasa de aprendizaje
        if new_z < current_z:
            learning_rate *= 0.8  # Reducir 20%
            if learning_rate < 1e-8:
                print(f"Tasa de aprendizaje muy pequeña en iteración {i}")
                break
            continue
        else:
            # Si mejoramos, podemos aumentar ligeramente la tasa
            learning_rate = min(learning_rate * 1.01, initial_lr)
        
        # Actualizar posición
        x, y = x_new, y_new
        history_x.append(x)
        history_y.append(y)
        history_z.append(new_z)
        history_lr.append(learning_rate)
        
        # Verificar convergencia
        if len(history_z) > 1 and abs(history_z[-1] - history_z[-2]) < 1e-8:
            print(f"Convergencia en iteración {i}")
            break
        
        if i % 50 == 0:
            print(f"Iter {i}: ({x:.4f}, {y:.4f}), valor: {new_z:.6f}, lr: {learning_rate:.6f}")
    
    return x, y, euler_function(x, y), history_x, history_y, history_z, history_lr

def plot_euler_optimization(history_x, history_y, history_z, final_x, final_y, final_z, title_suffix=""):
    """
    Grafica los resultados de la optimización de la función de Euler
    """
    # Crear malla para la superficie
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = euler_function(X, Y)
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Subplot 1: Vista 3D de la superficie con trayectoria
    ax1 = fig.add_subplot(131, projection='3d')
    surface = ax1.plot_surface(X, Y, Z, alpha=0.7, cmap='plasma', linewidth=0, antialiased=True)
    
    # Trayectoria del algoritmo
    ax1.plot(history_x, history_y, history_z, 'ro-', linewidth=3, markersize=5, 
             label='Trayectoria', alpha=0.9)
    
    # Punto inicial y final
    ax1.scatter([history_x[0]], [history_y[0]], [history_z[0]], 
               color='green', s=150, label='Inicio', marker='s')
    ax1.scatter([final_x], [final_y], [final_z], 
               color='red', s=150, label=f'Máximo: {final_z:.4f}', marker='*')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (Función de Euler)')
    ax1.set_title(f'Función de Euler 3D{title_suffix}')
    ax1.legend()
    
    # Subplot 2: Vista de contorno con trayectoria
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(X, Y, Z, levels=30, cmap='plasma', alpha=0.8)
    contour_lines = ax2.contour(X, Y, Z, levels=15, colors='black', alpha=0.4, linewidths=0.5)
    ax2.clabel(contour_lines, inline=True, fontsize=8)
    
    # Trayectoria
    ax2.plot(history_x, history_y, 'wo-', linewidth=2, markersize=4, 
             markeredgecolor='black', label='Trayectoria')
    ax2.scatter([history_x[0]], [history_y[0]], color='green', s=100, 
               label='Inicio', marker='s', edgecolor='black')
    ax2.scatter([final_x], [final_y], color='red', s=150, 
               label=f'Máximo: ({final_x:.3f}, {final_y:.3f})', marker='*', edgecolor='black')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Curvas de Nivel{title_suffix}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(contour, ax=ax2)
    
    # Subplot 3: Convergencia del valor de la función
    ax3 = fig.add_subplot(133)
    iterations = range(len(history_z))
    ax3.plot(iterations, history_z, 'b-', linewidth=2, marker='o', markersize=3)
    ax3.axhline(y=final_z, color='red', linestyle='--', alpha=0.7, 
               label=f'Valor final: {final_z:.6f}')
    ax3.set_xlabel('Iteración')
    ax3.set_ylabel('Valor de la Función')
    ax3.set_title(f'Convergencia{title_suffix}')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig

def run_euler_optimization():
    """
    Ejecuta el algoritmo desde múltiples puntos iniciales
    """
    print("🎯 ALGORITMO DE GRADIENTE ASCENDENTE PARA FUNCIÓN DE EULER")
    print("=" * 70)
    
    # Puntos iniciales estratégicos
    start_points = [
        (-2, -2, "Esquina inferior izquierda"),
        (2, 1, "Cerca del pico principal"),
        (-1, -1, "Centro-izquierda"),
        (0, 0, "Origen"),
        (3, -1, "Lado derecho"),
        (-3, 2, "Lado izquierdo superior")
    ]
    
    results = []
    
    print("\n🔍 OPTIMIZACIÓN ESTÁNDAR:")
    print("-" * 50)
    
    for i, (start_x, start_y, description) in enumerate(start_points):
        print(f"\n📍 Punto {i+1}: {description}")
        print(f"   Coordenadas iniciales: ({start_x}, {start_y})")
        
        final_x, final_y, final_z, hist_x, hist_y, hist_z = gradient_ascent_euler(
            start_x, start_y, learning_rate=0.05, max_iterations=500
        )
        
        results.append((final_x, final_y, final_z, hist_x, hist_y, hist_z, description))
        print(f"   ✅ Resultado: ({final_x:.6f}, {final_y:.6f}) = {final_z:.6f}")
    
    # Encontrar el mejor resultado
    best_idx = np.argmax([result[2] for result in results])
    best_result = results[best_idx]
    
    print(f"\n🏆 MEJOR RESULTADO ENCONTRADO:")
    print("=" * 50)
    print(f"Descripción: {best_result[6]}")
    print(f"Coordenadas: ({best_result[0]:.6f}, {best_result[1]:.6f})")
    print(f"Valor máximo: {best_result[2]:.6f}")
    print(f"Iteraciones: {len(best_result[3])}")
    
    # Graficar el mejor resultado
    plot_euler_optimization(best_result[3], best_result[4], best_result[5], 
                          best_result[0], best_result[1], best_result[2], 
                          " - Mejor Resultado")
    
    # Probar algoritmo adaptativo desde el mejor punto inicial
    print(f"\n🧠 OPTIMIZACIÓN ADAPTATIVA:")
    print("-" * 50)
    
    start_x, start_y = start_points[best_idx][:2]
    final_x_adap, final_y_adap, final_z_adap, hist_x_adap, hist_y_adap, hist_z_adap, hist_lr = \
        adaptive_gradient_ascent(start_x, start_y, initial_lr=0.1, max_iterations=300)
    
    print(f"✅ Resultado adaptativo: ({final_x_adap:.6f}, {final_y_adap:.6f}) = {final_z_adap:.6f}")
    
    # Graficar resultado adaptativo
    plot_euler_optimization(hist_x_adap, hist_y_adap, hist_z_adap, 
                          final_x_adap, final_y_adap, final_z_adap, 
                          " - Algoritmo Adaptativo")
    
    # Gráfica adicional: evolución de la tasa de aprendizaje
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(hist_lr, 'g-', linewidth=2, marker='o', markersize=3)
    plt.xlabel('Iteración')
    plt.ylabel('Tasa de Aprendizaje')
    plt.title('Evolución de la Tasa de Aprendizaje')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(hist_z_adap, 'b-', linewidth=2, marker='o', markersize=3)
    plt.xlabel('Iteración')
    plt.ylabel('Valor de la Función')
    plt.title('Convergencia del Algoritmo Adaptativo')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results, (final_x_adap, final_y_adap, final_z_adap)

# Función para comparar diferentes configuraciones
def compare_learning_rates():
    """
    Compara diferentes tasas de aprendizaje
    """
    print("\n📊 COMPARACIÓN DE TASAS DE APRENDIZAJE:")
    print("=" * 50)
    
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
    start_x, start_y = 2, 1  # Punto fijo para comparación
    
    plt.figure(figsize=(15, 10))
    
    for i, lr in enumerate(learning_rates):
        print(f"\n🔧 Tasa de aprendizaje: {lr}")
        final_x, final_y, final_z, hist_x, hist_y, hist_z = gradient_ascent_euler(
            start_x, start_y, learning_rate=lr, max_iterations=200
        )
        
        # Subplot para cada tasa de aprendizaje
        plt.subplot(2, 3, i+1)
        
        # Crear malla pequeña para el fondo
        x_range = np.linspace(-1, 4, 50)
        y_range = np.linspace(-1, 3, 50)
        X, Y = np.meshgrid(x_range, y_range)
        Z = euler_function(X, Y)
        
        plt.contourf(X, Y, Z, levels=20, alpha=0.6, cmap='plasma')
        plt.plot(hist_x, hist_y, 'wo-', linewidth=2, markersize=4, 
                markeredgecolor='black')
        plt.scatter([start_x], [start_y], color='green', s=100, marker='s', 
                   edgecolor='black', label='Inicio')
        plt.scatter([final_x], [final_y], color='red', s=100, marker='*', 
                   edgecolor='black', label=f'Final: {final_z:.3f}')
        
        plt.title(f'LR = {lr}\nIteraciones: {len(hist_x)}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Ejecutar el programa principal
if __name__ == "__main__":
    # Ejecutar optimización principal
    results, adaptive_result = run_euler_optimization()
    
    # Comparar tasas de aprendizaje
    compare_learning_rates()
    
    print(f"\n🎉 ¡OPTIMIZACIÓN COMPLETADA!")
    print(f"La función de Euler fue maximizada exitosamente.")
    print(f"Mejor valor encontrado: {max([r[2] for r in results]):.6f}")
