import numpy as np
import matplotlib.pyplot as plt
import random

def funcion_objetivo(x):
    """
    Función a MINIMIZAR (puede ser cualquier función)
    Ejemplo: función con múltiples mínimos locales
    """
    return x**2 * np.sin(5 * x) + x * np.cos(3 * x)

def generar_vecino(x_actual, temperatura):
    """
    Genera una solución vecina
    """
    # Perturbación proporcional a la temperatura
    perturbacion = np.random.uniform(-1, 1) * temperatura * 0.1
    return x_actual + perturbacion

def salto_aleatorio(limite_inferior, limite_superior):
    """
    Genera una solución completamente aleatoria (SALTO)
    """
    return np.random.uniform(limite_inferior, limite_superior)

def aceptar_solucion(costo_actual, costo_vecino, temperatura):
    """
    Decide si acepta la nueva solución
    """
    # Si es mejor, siempre acepta
    if costo_vecino < costo_actual:
        return True
    
    # Si es peor, acepta con probabilidad
    delta = costo_vecino - costo_actual
    probabilidad = np.exp(-delta / temperatura)
    return random.random() < probabilidad

def temple_simulado_simple():
    """
    Temple Simulado con Saltos Aleatorios - VERSIÓN DIDÁCTICA
    """
    print("="*60)
    print("TEMPLE SIMULADO CON SALTOS ALEATORIOS")
    print("="*60)
    
    # ========== PARÁMETROS ==========
    temperatura_inicial = 100
    temperatura_minima = 0.1
    factor_enfriamiento = 0.95
    max_iteraciones = 200
    prob_salto = 0.05  # 5% de probabilidad de salto
    
    # Límites del espacio de búsqueda
    limite_inferior = -5
    limite_superior = 5
    # ================================
    
    # Solución inicial aleatoria
    x_actual = np.random.uniform(limite_inferior, limite_superior)
    costo_actual = funcion_objetivo(x_actual)
    
    # Mejor solución encontrada
    mejor_x = x_actual
    mejor_costo = costo_actual
    
    # Para graficar
    historial_x = [x_actual]
    historial_costo = [costo_actual]
    historial_mejor_costo = [mejor_costo]
    historial_temperatura = [temperatura_inicial]
    historial_saltos = []
    
    temperatura = temperatura_inicial
    iteracion = 0
    num_saltos = 0
    
    print(f"\nSolución inicial: x = {x_actual:.4f}, costo = {costo_actual:.4f}")
    print(f"\nIniciando optimización...\n")
    
    # ========== BUCLE PRINCIPAL ==========
    while temperatura > temperatura_minima and iteracion < max_iteraciones:
        
        # Decidir si hacer un SALTO ALEATORIO
        if random.random() < prob_salto:
            x_vecino = salto_aleatorio(limite_inferior, limite_superior)
            num_saltos += 1
            historial_saltos.append(iteracion)
            print(f"🚀 SALTO ALEATORIO en iteración {iteracion}")
        else:
            # Generar vecino normal
            x_vecino = generar_vecino(x_actual, temperatura)
            # Mantener dentro de límites
            x_vecino = np.clip(x_vecino, limite_inferior, limite_superior)
        
        # Evaluar vecino
        costo_vecino = funcion_objetivo(x_vecino)
        
        # Decidir si aceptar
        if aceptar_solucion(costo_actual, costo_vecino, temperatura):
            x_actual = x_vecino
            costo_actual = costo_vecino
            
            # Actualizar mejor solución
            if costo_actual < mejor_costo:
                mejor_x = x_actual
                mejor_costo = costo_actual
                print(f"✓ Iter {iteracion:3d}: Nueva mejor solución! "
                      f"x = {mejor_x:.4f}, costo = {mejor_costo:.4f}")
        
        # Guardar historial
        historial_x.append(x_actual)
        historial_costo.append(costo_actual)
        historial_mejor_costo.append(mejor_costo)
        historial_temperatura.append(temperatura)
        
        # Enfriar
        temperatura *= factor_enfriamiento
        iteracion += 1
    
    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL")
    print(f"{'='*60}")
    print(f"Mejor solución: x = {mejor_x:.4f}")
    print(f"Mejor costo: {mejor_costo:.4f}")
    print(f"Iteraciones: {iteracion}")
    print(f"Saltos aleatorios: {num_saltos}")
    print(f"{'='*60}\n")
    
    # ========== GRAFICAR RESULTADOS ==========
    graficar_resultados(historial_x, historial_costo, historial_mejor_costo,
                       historial_temperatura, historial_saltos,
                       mejor_x, mejor_costo, limite_inferior, limite_superior)

def graficar_resultados(historial_x, historial_costo, historial_mejor_costo,
                       historial_temperatura, historial_saltos,
                       mejor_x, mejor_costo, lim_inf, lim_sup):
    """
    Grafica los resultados del algoritmo
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TEMPLE SIMULADO CON SALTOS ALEATORIOS', 
                 fontsize=16, fontweight='bold')
    
    # ========== GRÁFICA 1: Función Objetivo ==========
    ax1 = axes[0, 0]
    x_plot = np.linspace(lim_inf, lim_sup, 1000)
    y_plot = funcion_objetivo(x_plot)
    
    ax1.plot(x_plot, y_plot, 'b-', linewidth=2, label='Función Objetivo')
    ax1.plot(mejor_x, mejor_costo, 'r*', markersize=20, 
            label=f'Mejor Solución\nx={mejor_x:.3f}', zorder=5)
    
    # Marcar el recorrido
    ax1.plot(historial_x, [funcion_objetivo(x) for x in historial_x],
            'go-', alpha=0.3, markersize=3, label='Recorrido')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Función Objetivo y Búsqueda', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== GRÁFICA 2: Convergencia del Costo ==========
    ax2 = axes[0, 1]
    iteraciones = range(len(historial_costo))
    
    ax2.plot(iteraciones, historial_costo, 'b-', alpha=0.5, 
            linewidth=1, label='Costo Actual')
    ax2.plot(iteraciones, historial_mejor_costo, 'r-', 
            linewidth=2, label='Mejor Costo')
    
    # Marcar saltos aleatorios
    if historial_saltos:
        for salto_iter in historial_saltos:
            ax2.axvline(x=salto_iter, color='green', linestyle='--', 
                       alpha=0.5, linewidth=1)
        ax2.plot([], [], 'g--', label='Saltos Aleatorios')
    
    ax2.set_xlabel('Iteración', fontsize=12)
    ax2.set_ylabel('Costo', fontsize=12)
    ax2.set_title('Convergencia del Algoritmo', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ========== GRÁFICA 3: Temperatura ==========
    ax3 = axes[1, 0]
    ax3.plot(iteraciones, historial_temperatura, 'orange', linewidth=2)
    ax3.set_xlabel('Iteración', fontsize=12)
    ax3.set_ylabel('Temperatura', fontsize=12)
    ax3.set_title('Enfriamiento del Sistema', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # ========== GRÁFICA 4: Exploración del Espacio ==========
    ax4 = axes[1, 1]
    scatter = ax4.scatter(iteraciones, historial_x, 
                         c=historial_temperatura, cmap='hot',
                         s=30, alpha=0.6)
    ax4.axhline(y=mejor_x, color='red', linestyle='--', 
               linewidth=2, label=f'Mejor x = {mejor_x:.3f}')
    
    # Marcar saltos
    if historial_saltos:
        saltos_x = [historial_x[i] for i in historial_saltos]
        ax4.scatter(historial_saltos, saltos_x, color='green', 
                   s=100, marker='^', label='Saltos', zorder=5)
    
    ax4.set_xlabel('Iteración', fontsize=12)
    ax4.set_ylabel('Posición x', fontsize=12)
    ax4.set_title('Exploración del Espacio de Búsqueda', 
                 fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Barra de color
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Temperatura', fontsize=11)
    
    plt.tight_layout()
    plt.show()

# ========== EJECUTAR ==========
if __name__ == "__main__":
    temple_simulado_simple()
