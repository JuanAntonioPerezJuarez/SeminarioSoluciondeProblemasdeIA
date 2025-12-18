import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class PSO_SegmentacionMultinivel:
    """
    Segmentación multinivel de imágenes usando Particle Swarm Optimization (PSO)
    """
    
    def __init__(self, imagen_ruta, num_umbrales=3):
        """
        Inicializa el algoritmo PSO para segmentación
        
        Args:
            imagen_ruta: Ruta de la imagen a segmentar
            num_umbrales: Número de umbrales a encontrar (niveles = umbrales + 1)
        """
        self.imagen_original = cv2.imread(imagen_ruta)
        if self.imagen_original is None:
            raise ValueError("No se pudo cargar la imagen")
        
        # Convertir a escala de grises
        self.imagen_gris = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)
        self.num_umbrales = num_umbrales
        
        # Calcular histograma
        self.histograma = cv2.calcHist([self.imagen_gris], [0], None, [256], [0, 256])
        self.histograma = self.histograma.flatten() / self.histograma.sum()
        
        print(f"✓ Imagen cargada: {self.imagen_gris.shape}")
        print(f"✓ Número de umbrales a buscar: {num_umbrales}")
        print(f"✓ Número de niveles resultantes: {num_umbrales + 1}")
    
    def calcular_varianza_entre_clases(self, umbrales):
        """
        Calcula la varianza entre clases (método de Otsu multinivel)
        Esta es la función objetivo a MAXIMIZAR
        
        Args:
            umbrales: Lista de umbrales a evaluar
        
        Returns:
            Varianza entre clases
        """
        umbrales = sorted(umbrales)
        umbrales = [0] + umbrales + [256]
        
        varianza_total = 0
        media_global = sum(i * self.histograma[i] for i in range(256))
        
        for i in range(len(umbrales) - 1):
            inicio = int(umbrales[i])
            fin = int(umbrales[i + 1])
            
            # Probabilidad de la clase
            w = sum(self.histograma[inicio:fin])
            
            if w > 0:
                # Media de la clase
                media_clase = sum(j * self.histograma[j] for j in range(inicio, fin)) / w
                # Contribución a la varianza
                varianza_total += w * (media_clase - media_global) ** 2
        
        return varianza_total
    
    def calcular_entropia(self, umbrales):
        """
        Calcula la entropía de Kapur (método alternativo)
        Esta es la función objetivo a MAXIMIZAR
        
        Args:
            umbrales: Lista de umbrales a evaluar
        
        Returns:
            Entropía total
        """
        umbrales = sorted(umbrales)
        umbrales = [0] + umbrales + [256]
        
        entropia_total = 0
        
        for i in range(len(umbrales) - 1):
            inicio = int(umbrales[i])
            fin = int(umbrales[i + 1])
            
            # Probabilidad de la clase
            w = sum(self.histograma[inicio:fin])
            
            if w > 0:
                # Entropía de la clase
                entropia_clase = 0
                for j in range(inicio, fin):
                    if self.histograma[j] > 0:
                        p = self.histograma[j] / w
                        entropia_clase -= p * np.log2(p + 1e-10)
                
                entropia_total += entropia_clase
        
        return entropia_total
    
    def optimizar_pso(self, num_particulas=30, num_iteraciones=100, 
                      metodo='varianza', w=0.7, c1=1.5, c2=1.5):
        """
        Ejecuta el algoritmo PSO para encontrar los umbrales óptimos
        
        Args:
            num_particulas: Número de partículas en el enjambre
            num_iteraciones: Número de iteraciones
            metodo: 'varianza' o 'entropia'
            w: Peso de inercia
            c1: Coeficiente cognitivo
            c2: Coeficiente social
        
        Returns:
            Mejores umbrales encontrados
        """
        print(f"\n{'='*60}")
        print(f"Ejecutando PSO - Método: {metodo.upper()}")
        print(f"{'='*60}")
        
        # Inicializar partículas (posiciones aleatorias)
        particulas = np.random.uniform(1, 255, (num_particulas, self.num_umbrales))
        velocidades = np.random.uniform(-10, 10, (num_particulas, self.num_umbrales))
        
        # Mejor posición personal de cada partícula
        mejor_personal = particulas.copy()
        mejor_fitness_personal = np.array([self._evaluar_fitness(p, metodo) 
                                           for p in particulas])
        
        # Mejor posición global
        idx_mejor_global = np.argmax(mejor_fitness_personal)
        mejor_global = mejor_personal[idx_mejor_global].copy()
        mejor_fitness_global = mejor_fitness_personal[idx_mejor_global]
        
        # Historial de fitness
        historial_fitness = []
        
        # Iteraciones PSO
        for iteracion in range(num_iteraciones):
            for i in range(num_particulas):
                # Actualizar velocidad
                r1, r2 = np.random.rand(2)
                velocidades[i] = (w * velocidades[i] + 
                                 c1 * r1 * (mejor_personal[i] - particulas[i]) +
                                 c2 * r2 * (mejor_global - particulas[i]))
                
                # Actualizar posición
                particulas[i] = particulas[i] + velocidades[i]
                
                # Mantener dentro de límites [1, 255]
                particulas[i] = np.clip(particulas[i], 1, 255)
                
                # Evaluar fitness
                fitness = self._evaluar_fitness(particulas[i], metodo)
                
                # Actualizar mejor personal
                if fitness > mejor_fitness_personal[i]:
                    mejor_fitness_personal[i] = fitness
                    mejor_personal[i] = particulas[i].copy()
                    
                    # Actualizar mejor global
                    if fitness > mejor_fitness_global:
                        mejor_fitness_global = fitness
                        mejor_global = particulas[i].copy()
            
            historial_fitness.append(mejor_fitness_global)
            
            # Mostrar progreso cada 10 iteraciones
            if (iteracion + 1) % 10 == 0:
                print(f"Iteración {iteracion + 1}/{num_iteraciones} - "
                      f"Mejor fitness: {mejor_fitness_global:.6f}")
        
        # Ordenar umbrales
        mejor_global = sorted(mejor_global)
        
        print(f"\n{'='*60}")
        print(f"✓ Optimización completada")
        print(f"✓ Umbrales óptimos encontrados: {[int(u) for u in mejor_global]}")
        print(f"✓ Fitness final: {mejor_fitness_global:.6f}")
        print(f"{'='*60}\n")
        
        # Graficar convergencia
        self._graficar_convergencia(historial_fitness)
        
        return mejor_global, historial_fitness
    
    def _evaluar_fitness(self, umbrales, metodo):
        """
        Evalúa el fitness de una solución
        """
        if metodo == 'varianza':
            return self.calcular_varianza_entre_clases(umbrales)
        elif metodo == 'entropia':
            return self.calcular_entropia(umbrales)
        else:
            raise ValueError("Método debe ser 'varianza' o 'entropia'")
    
    def segmentar_imagen(self, umbrales):
        """
        Segmenta la imagen usando los umbrales encontrados
        
        Args:
            umbrales: Lista de umbrales
        
        Returns:
            Imagen segmentada
        """
        umbrales = sorted(umbrales)
        imagen_segmentada = np.zeros_like(self.imagen_gris)
        
        # Asignar niveles
        nivel = 0
        imagen_segmentada[self.imagen_gris <= umbrales[0]] = nivel
        
        for i in range(len(umbrales) - 1):
            nivel += 1
            mascara = (self.imagen_gris > umbrales[i]) & (self.imagen_gris <= umbrales[i + 1])
            imagen_segmentada[mascara] = nivel
        
        nivel += 1
        imagen_segmentada[self.imagen_gris > umbrales[-1]] = nivel
        
        # Normalizar para visualización
        imagen_segmentada = (imagen_segmentada * (255 // (len(umbrales)))).astype(np.uint8)
        
        return imagen_segmentada
    
    def _graficar_convergencia(self, historial_fitness):
        """
        Grafica la convergencia del algoritmo PSO
        """
        plt.figure(figsize=(10, 5))
        plt.plot(historial_fitness, linewidth=2, color='blue')
        plt.xlabel('Iteración', fontsize=12)
        plt.ylabel('Mejor Fitness', fontsize=12)
        plt.title('Convergencia del Algoritmo PSO', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def visualizar_resultados(self, umbrales, imagen_segmentada):
        """
        Visualiza los resultados de la segmentación
        """
        fig = plt.figure(figsize=(18, 10))
        
        # Imagen original
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2RGB))
        plt.title('Imagen Original', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Imagen en escala de grises
        plt.subplot(2, 3, 2)
        plt.imshow(self.imagen_gris, cmap='gray')
        plt.title('Escala de Grises', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Imagen segmentada
        plt.subplot(2, 3, 3)
        plt.imshow(imagen_segmentada, cmap='jet')
        plt.title(f'Imagen Segmentada ({len(umbrales)+1} niveles)', 
                 fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Histograma con umbrales
        plt.subplot(2, 3, 4)
        plt.plot(self.histograma * 1000, color='black', linewidth=1.5)
        plt.fill_between(range(256), self.histograma * 1000, alpha=0.3)
        
        colores = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
        for i, umbral in enumerate(umbrales):
            color = colores[i % len(colores)]
            plt.axvline(x=umbral, color=color, linestyle='--', linewidth=2, 
                       label=f'Umbral {i+1}: {int(umbral)}')
        
        plt.xlabel('Intensidad de Píxel', fontsize=11)
        plt.ylabel('Frecuencia Normalizada (×1000)', fontsize=11)
        plt.title('Histograma con Umbrales Óptimos', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Segmentación con colores personalizados
        plt.subplot(2, 3, 5)
        plt.imshow(imagen_segmentada, cmap='viridis')
        plt.title('Segmentación (Viridis)', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Segmentación con otro mapa de colores
        plt.subplot(2, 3, 6)
        plt.imshow(imagen_segmentada, cmap='plasma')
        plt.title('Segmentación (Plasma)', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def guardar_resultados(self, imagen_segmentada, nombre_base='segmentada'):
        """
        Guarda la imagen segmentada
        """
        cv2.imwrite(f'{nombre_base}.jpg', imagen_segmentada)
        print(f"✓ Imagen segmentada guardada como '{nombre_base}.jpg'")


def ejecutar_segmentacion_completa(ruta_imagen, num_umbrales=3, metodo='varianza'):
    """
    Función principal para ejecutar la segmentación completa
    
    Args:
        ruta_imagen: Ruta de la imagen
        num_umbrales: Número de umbrales (2-5 recomendado)
        metodo: 'varianza' o 'entropia'
    """
    try:
        # Crear objeto de segmentación
        segmentador = PSO_SegmentacionMultinivel(ruta_imagen, num_umbrales)
        
        # Ejecutar PSO
        umbrales_optimos, historial = segmentador.optimizar_pso(
            num_particulas=30,
            num_iteraciones=100,
            metodo=metodo,
            w=0.7,      # Inercia
            c1=1.5,     # Componente cognitivo
            c2=1.5      # Componente social
        )
        
        # Segmentar imagen
        imagen_segmentada = segmentador.segmentar_imagen(umbrales_optimos)
        
        # Visualizar resultados
        segmentador.visualizar_resultados(umbrales_optimos, imagen_segmentada)
        
        # Guardar resultados
        segmentador.guardar_resultados(imagen_segmentada, 
                                       f'segmentada_{num_umbrales}niveles')
        
        return segmentador, umbrales_optimos, imagen_segmentada
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None


# EJEMPLO DE USO
if __name__ == "__main__":
    print("="*70)
    print(" SEGMENTACIÓN MULTINIVEL DE IMÁGENES USANDO PSO")
    print("="*70)
    
    # ========== CONFIGURA AQUÍ ==========
    ruta_imagen = '../ActividadClase/example.jpg'  # <-- CAMBIA ESTO por tu imagen
    num_umbrales = 3                # Número de umbrales (2-5 recomendado)
    metodo = 'varianza'             # 'varianza' o 'entropia'
    # ====================================
    
    # Ejecutar segmentación
    segmentador, umbrales, img_segmentada = ejecutar_segmentacion_completa(
        ruta_imagen, 
        num_umbrales, 
        metodo
    )
    
    print("\n" + "="*70)
    print("✓ PROCESO COMPLETADO")
    print("="*70)
