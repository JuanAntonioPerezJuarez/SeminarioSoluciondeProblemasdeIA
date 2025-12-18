import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from skimage.metrics import structural_similarity as ssim

class TempleSimuladoConSaltos:
    """
    Algoritmo de Temple Simulado con Saltos Aleatorios para Segmentación Multinivel
    
    El temple simulado es una metaheurística inspirada en el proceso de recocido 
    de metales. Los saltos aleatorios ayudan a escapar de óptimos locales.
    """
    
    def __init__(self, imagen_ruta, num_umbrales=3):
        """
        Inicializa el algoritmo de Temple Simulado
        
        Args:
            imagen_ruta: Ruta de la imagen a segmentar
            num_umbrales: Número de umbrales a encontrar
        """
        # Cargar imagen
        self.imagen_original = cv2.imread(imagen_ruta)
        if self.imagen_original is None:
            raise ValueError("No se pudo cargar la imagen")
        
        # Convertir a escala de grises
        self.imagen_gris = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)
        self.num_umbrales = num_umbrales
        
        # Calcular histograma normalizado
        self.histograma = cv2.calcHist([self.imagen_gris], [0], None, [256], [0, 256])
        self.histograma = self.histograma.flatten() / self.histograma.sum()
        
        # Variables para tracking
        self.historial_energia = []
        self.historial_temperatura = []
        self.historial_mejor_energia = []
        self.historial_saltos = []
        self.contador_saltos = 0
        
        # Variables para métricas de calidad
        self.metricas_calidad = {}
        
        print(f"✓ Imagen cargada: {self.imagen_gris.shape}")
        print(f"✓ Número de umbrales: {num_umbrales}")
        print(f"✓ Número de niveles: {num_umbrales + 1}")
    
    def calcular_mse(self, imagen_original, imagen_segmentada):
        """
        Calcula el Error Cuadrático Medio (MSE - Mean Squared Error)
        
        MSE = (1/N) * Σ(I_original - I_segmentada)²
        
        Menor MSE = Mayor similitud
        
        Args:
            imagen_original: Imagen original en escala de grises
            imagen_segmentada: Imagen segmentada
        
        Returns:
            Valor MSE
        """
        mse = np.mean((imagen_original.astype(float) - imagen_segmentada.astype(float)) ** 2)
        return mse
    
    def calcular_psnr(self, imagen_original, imagen_segmentada):
        """
        Calcula la Relación Señal-Ruido de Pico (PSNR - Peak Signal-to-Noise Ratio)
        
        PSNR = 10 * log10(MAX² / MSE)
        donde MAX = 255 para imágenes de 8 bits
        
        Mayor PSNR = Mejor calidad (típicamente > 30 dB es bueno)
        
        Args:
            imagen_original: Imagen original en escala de grises
            imagen_segmentada: Imagen segmentada
        
        Returns:
            Valor PSNR en dB
        """
        mse = self.calcular_mse(imagen_original, imagen_segmentada)
        
        if mse == 0:
            return float('inf')  # Imágenes idénticas
        
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        
        return psnr
    
    def calcular_ssim(self, imagen_original, imagen_segmentada):
        """
        Calcula el Índice de Similitud Estructural (SSIM - Structural Similarity Index)
        
        SSIM considera luminancia, contraste y estructura
        Rango: [-1, 1], donde 1 = imágenes idénticas
        
        SSIM > 0.9 = Excelente
        SSIM > 0.8 = Buena
        SSIM > 0.7 = Aceptable
        
        Args:
            imagen_original: Imagen original en escala de grises
            imagen_segmentada: Imagen segmentada
        
        Returns:
            Valor SSIM
        """
        # Calcular SSIM usando scikit-image
        ssim_value = ssim(imagen_original, imagen_segmentada, data_range=255)
        
        return ssim_value
    
    def calcular_todas_metricas(self, imagen_segmentada):
        """
        Calcula todas las métricas de calidad
        
        Args:
            imagen_segmentada: Imagen segmentada
        
        Returns:
            Diccionario con todas las métricas
        """
        mse = self.calcular_mse(self.imagen_gris, imagen_segmentada)
        psnr = self.calcular_psnr(self.imagen_gris, imagen_segmentada)
        ssim_value = self.calcular_ssim(self.imagen_gris, imagen_segmentada)
        
        metricas = {
            'MSE': mse,
            'PSNR': psnr,
            'SSIM': ssim_value
        }
        
        self.metricas_calidad = metricas
        
        return metricas
    
    def mostrar_metricas(self, metricas):
        """
        Muestra las métricas de calidad de forma formateada
        
        Args:
            metricas: Diccionario con las métricas
        """
        print(f"\n{'='*70}")
        print(f"MÉTRICAS DE CALIDAD DE SEGMENTACIÓN")
        print(f"{'='*70}")
        print(f"\n   MSE (Error Cuadrático Medio)")
        print(f"     Valor: {metricas['MSE']:.4f}")
        print(f"     Interpretación: Menor es mejor (0 = idéntico)")
        
        print(f"\n   PSNR (Relación Señal-Ruido de Pico)")
        print(f"     Valor: {metricas['PSNR']:.4f} dB")
        if metricas['PSNR'] > 40:
            calidad_psnr = "Excelente"
        elif metricas['PSNR'] > 30:
            calidad_psnr = "Buena"
        elif metricas['PSNR'] > 20:
            calidad_psnr = "Aceptable"
        else:
            calidad_psnr = "Baja"
        print(f"     Interpretación: {calidad_psnr} (>30 dB es bueno)")
        
        print(f"\n   SSIM (Índice de Similitud Estructural)")
        print(f"     Valor: {metricas['SSIM']:.4f}")
        if metricas['SSIM'] > 0.9:
            calidad_ssim = "Excelente"
        elif metricas['SSIM'] > 0.8:
            calidad_ssim = "Buena"
        elif metricas['SSIM'] > 0.7:
            calidad_ssim = "Aceptable"
        else:
            calidad_ssim = "Baja"
        print(f"     Interpretación: {calidad_ssim} (1 = idéntico)")
        
        print(f"\n{'='*70}\n")
    
    def calcular_energia(self, umbrales):
        """
        Calcula la energía de una solución (función objetivo)
        Usamos la varianza entre clases de Otsu (queremos MAXIMIZAR)
        Por eso retornamos el negativo (para MINIMIZAR energía)
        
        Args:
            umbrales: Lista de umbrales
        
        Returns:
            Energía (negativo de la varianza entre clases)
        """
        umbrales = sorted(umbrales)
        umbrales_completos = [0] + umbrales + [256]
        
        varianza_total = 0
        media_global = sum(i * self.histograma[i] for i in range(256))
        
        # Calcular varianza entre clases
        for i in range(len(umbrales_completos) - 1):
            inicio = int(umbrales_completos[i])
            fin = int(umbrales_completos[i + 1])
            
            # Probabilidad de la clase
            w = sum(self.histograma[inicio:fin])
            
            if w > 0:
                # Media de la clase
                media_clase = sum(j * self.histograma[j] for j in range(inicio, fin)) / w
                # Contribución a la varianza
                varianza_total += w * (media_clase - media_global) ** 2
        
        # Retornar negativo (queremos minimizar energía = maximizar varianza)
        return -varianza_total
    
    def generar_vecino(self, solucion_actual, temperatura):
        """
        Genera una solución vecina mediante perturbación
        La magnitud de la perturbación depende de la temperatura
        
        Args:
            solucion_actual: Solución actual
            temperatura: Temperatura actual (afecta la magnitud del cambio)
        
        Returns:
            Nueva solución vecina
        """
        vecino = solucion_actual.copy()
        
        # Seleccionar un umbral aleatorio para perturbar
        idx = np.random.randint(0, len(vecino))
        
        # Perturbación proporcional a la temperatura
        # A mayor temperatura, mayor exploración
        perturbacion = np.random.normal(0, temperatura * 0.1)
        vecino[idx] += perturbacion
        
        # Mantener dentro de límites [1, 255]
        vecino[idx] = np.clip(vecino[idx], 1, 255)
        
        return vecino
    
    def realizar_salto_aleatorio(self):
        """
        Realiza un salto aleatorio a una región completamente diferente
        del espacio de búsqueda para escapar de óptimos locales
        
        Returns:
            Nueva solución aleatoria
        """
        self.contador_saltos += 1
        print(f" SALTO ALEATORIO #{self.contador_saltos} - Escapando de óptimo local")
        
        # Generar solución completamente nueva
        nueva_solucion = np.random.uniform(1, 255, self.num_umbrales)
        return nueva_solucion
    
    def criterio_aceptacion(self, energia_actual, energia_vecino, temperatura):
        """
        Criterio de aceptación de Metrópolis
        
        - Si la nueva solución es mejor (menor energía), siempre se acepta
        - Si es peor, se acepta con probabilidad exp(-ΔE/T)
        
        Args:
            energia_actual: Energía de la solución actual
            energia_vecino: Energía de la solución vecina
            temperatura: Temperatura actual
        
        Returns:
            True si se acepta la solución vecina, False en caso contrario
        """
        # Si mejora, siempre aceptar
        if energia_vecino < energia_actual:
            return True
        
        # Si empeora, aceptar con cierta probabilidad
        delta_energia = energia_vecino - energia_actual
        probabilidad = np.exp(-delta_energia / temperatura)
        
        return np.random.random() < probabilidad
    
    def enfriamiento_exponencial(self, temperatura, alpha=0.95):
        """
        Esquema de enfriamiento exponencial
        T(k+1) = alpha * T(k)
        
        Args:
            temperatura: Temperatura actual
            alpha: Factor de enfriamiento (0 < alpha < 1)
        
        Returns:
            Nueva temperatura
        """
        return temperatura * alpha
    
    def enfriamiento_logaritmico(self, temperatura_inicial, iteracion):
        """
        Esquema de enfriamiento logarítmico (más lento)
        T(k) = T0 / log(1 + k)
        
        Args:
            temperatura_inicial: Temperatura inicial
            iteracion: Número de iteración actual
        
        Returns:
            Nueva temperatura
        """
        return temperatura_inicial / np.log(2 + iteracion)
    
    def optimizar(self, temperatura_inicial=100, temperatura_final=0.01, 
                  alpha=0.95, iteraciones_por_temp=50, 
                  prob_salto=0.05, iteraciones_sin_mejora_max=200,
                  esquema_enfriamiento='exponencial'):
        """
        Ejecuta el algoritmo de Temple Simulado con Saltos Aleatorios
        
        Args:
            temperatura_inicial: Temperatura inicial (alta = más exploración)
            temperatura_final: Temperatura final (criterio de parada)
            alpha: Factor de enfriamiento (0.8-0.99)
            iteraciones_por_temp: Iteraciones en cada temperatura
            prob_salto: Probabilidad de realizar un salto aleatorio
            iteraciones_sin_mejora_max: Máximo de iteraciones sin mejora antes de salto forzado
            esquema_enfriamiento: 'exponencial' o 'logaritmico'
        
        Returns:
            Mejor solución encontrada
        """
        print(f"\n{'='*70}")
        print(f"TEMPLE SIMULADO CON SALTOS ALEATORIOS")
        print(f"{'='*70}")
        print(f"Parámetros:")
        print(f"  • Temperatura inicial: {temperatura_inicial}")
        print(f"  • Temperatura final: {temperatura_final}")
        print(f"  • Factor de enfriamiento (α): {alpha}")
        print(f"  • Iteraciones por temperatura: {iteraciones_por_temp}")
        print(f"  • Probabilidad de salto: {prob_salto}")
        print(f"  • Esquema de enfriamiento: {esquema_enfriamiento}")
        print(f"{'='*70}\n")
        
        # Inicializar solución aleatoria
        solucion_actual = np.random.uniform(1, 255, self.num_umbrales)
        energia_actual = self.calcular_energia(solucion_actual)
        
        # Mejor solución encontrada
        mejor_solucion = solucion_actual.copy()
        mejor_energia = energia_actual
        
        # Variables de control
        temperatura = temperatura_inicial
        iteracion_global = 0
        iteraciones_sin_mejora = 0
        
        # Limpiar historial
        self.historial_energia = []
        self.historial_temperatura = []
        self.historial_mejor_energia = []
        self.historial_saltos = []
        self.contador_saltos = 0
        
        tiempo_inicio = time.time()
        
        # Ciclo principal del temple simulado
        while temperatura > temperatura_final:
            
            for _ in range(iteraciones_por_temp):
                iteracion_global += 1
                iteraciones_sin_mejora += 1
                
                # MECANISMO DE SALTOS ALEATORIOS
                # Salto aleatorio probabilístico
                if np.random.random() < prob_salto:
                    solucion_vecina = self.realizar_salto_aleatorio()
                    self.historial_saltos.append(iteracion_global)
                
                # Salto forzado si hay estancamiento
                elif iteraciones_sin_mejora > iteraciones_sin_mejora_max:
                    print(f"    Estancamiento detectado ({iteraciones_sin_mejora} iter sin mejora)")
                    solucion_vecina = self.realizar_salto_aleatorio()
                    self.historial_saltos.append(iteracion_global)
                    iteraciones_sin_mejora = 0
                
                # Generación normal de vecino
                else:
                    solucion_vecina = self.generar_vecino(solucion_actual, temperatura)
                
                # Calcular energía del vecino
                energia_vecina = self.calcular_energia(solucion_vecina)
                
                # Criterio de aceptación
                if self.criterio_aceptacion(energia_actual, energia_vecina, temperatura):
                    solucion_actual = solucion_vecina
                    energia_actual = energia_vecina
                    
                    # Actualizar mejor solución
                    if energia_actual < mejor_energia:
                        mejor_solucion = solucion_actual.copy()
                        mejor_energia = energia_actual
                        iteraciones_sin_mejora = 0
                        print(f"  ✓ Nueva mejor solución encontrada: Energía = {-mejor_energia:.6f}")
                
                # Guardar historial
                self.historial_energia.append(energia_actual)
                self.historial_temperatura.append(temperatura)
                self.historial_mejor_energia.append(mejor_energia)
            
            # Enfriar temperatura
            if esquema_enfriamiento == 'exponencial':
                temperatura = self.enfriamiento_exponencial(temperatura, alpha)
            else:
                temperatura = self.enfriamiento_logaritmico(temperatura_inicial, iteracion_global)
            
            # Mostrar progreso
            if iteracion_global % (iteraciones_por_temp * 10) == 0:
                print(f"Iteración {iteracion_global} - Temp: {temperatura:.4f} - "
                      f"Mejor Energía: {-mejor_energia:.6f}")
        
        tiempo_total = time.time() - tiempo_inicio
        
        # Ordenar umbrales
        mejor_solucion = sorted(mejor_solucion)
        
        print(f"\n{'='*70}")
        print(f"✓ OPTIMIZACIÓN COMPLETADA")
        print(f"{'='*70}")
        print(f"  • Tiempo total: {tiempo_total:.2f} segundos")
        print(f"  • Iteraciones totales: {iteracion_global}")
        print(f"  • Saltos aleatorios realizados: {self.contador_saltos}")
        print(f"  • Umbrales óptimos: {[int(u) for u in mejor_solucion]}")
        print(f"  • Mejor energía: {-mejor_energia:.6f}")
        print(f"{'='*70}\n")
        
        return mejor_solucion
    
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
        imagen_segmentada = (imagen_segmentada * (255 // len(umbrales))).astype(np.uint8)
        
        return imagen_segmentada
    
    def visualizar_convergencia(self):
        """
        Visualiza la convergencia del algoritmo con gráficas detalladas
        """
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Energía vs Iteración
        ax1 = plt.subplot(3, 3, 1)
        iteraciones = range(len(self.historial_energia))
        ax1.plot(iteraciones, [-e for e in self.historial_energia], 
                 color='blue', alpha=0.6, linewidth=1, label='Energía actual')
        ax1.plot(iteraciones, [-e for e in self.historial_mejor_energia], 
                 color='red', linewidth=2, label='Mejor energía')
        
        # Marcar saltos aleatorios
        for salto in self.historial_saltos:
            ax1.axvline(x=salto, color='green', alpha=0.3, linestyle='--', linewidth=1)
        
        ax1.set_xlabel('Iteración', fontsize=11)
        ax1.set_ylabel('Varianza Entre Clases', fontsize=11)
        ax1.set_title('Convergencia del Algoritmo', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Temperatura vs Iteración
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(iteraciones, self.historial_temperatura, color='orange', linewidth=2)
        ax2.set_xlabel('Iteración', fontsize=11)
        ax2.set_ylabel('Temperatura', fontsize=11)
        ax2.set_title('Enfriamiento del Sistema', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. Energía vs Temperatura (Diagrama de fase)
        ax3 = plt.subplot(3, 3, 3)
        scatter = ax3.scatter(self.historial_temperatura, 
                             [-e for e in self.historial_energia],
                             c=range(len(self.historial_temperatura)), 
                             cmap='viridis', s=1, alpha=0.6)
        ax3.set_xlabel('Temperatura', fontsize=11)
        ax3.set_ylabel('Varianza Entre Clases', fontsize=11)
        ax3.set_title('Diagrama de Fase', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Iteración')
        
        # 4. Histograma de energías
        ax4 = plt.subplot(3, 3, 4)
        ax4.hist([-e for e in self.historial_energia], bins=50, 
                color='purple', alpha=0.7, edgecolor='black')
        ax4.axvline(x=-self.historial_mejor_energia[-1], color='red', 
                   linestyle='--', linewidth=2, label='Mejor solución')
        ax4.set_xlabel('Varianza Entre Clases', fontsize=11)
        ax4.set_ylabel('Frecuencia', fontsize=11)
        ax4.set_title('Distribución de Energías', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Tasa de aceptación por temperatura
        ax5 = plt.subplot(3, 3, 5)
        ventana = 100
        mejoras = []
        for i in range(0, len(self.historial_energia) - ventana, ventana):
            ventana_energia = self.historial_energia[i:i+ventana]
            tasa_mejora = sum(1 for j in range(1, len(ventana_energia)) 
                            if ventana_energia[j] < ventana_energia[j-1]) / ventana
            mejoras.append(tasa_mejora)
        
        ax5.plot(mejoras, color='green', linewidth=2, marker='o', markersize=4)
        ax5.set_xlabel('Ventana de iteraciones', fontsize=11)
        ax5.set_ylabel('Tasa de Mejora', fontsize=11)
        ax5.set_title('Tasa de Aceptación de Mejoras', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Saltos aleatorios
        ax6 = plt.subplot(3, 3, 6)
        if self.historial_saltos:
            ax6.scatter(self.historial_saltos, 
                       [1] * len(self.historial_saltos),
                       color='red', s=100, marker='*', 
                       label=f'Saltos ({len(self.historial_saltos)})')
            ax6.set_xlim(0, len(self.historial_energia))
        ax6.set_xlabel('Iteración', fontsize=11)
        ax6.set_title('Saltos Aleatorios Realizados', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_yticks([])
        
        # 7. Mejora acumulada
        ax7 = plt.subplot(3, 3, 7)
        mejora_acumulada = [(-self.historial_mejor_energia[0] - (-e)) 
                           for e in self.historial_mejor_energia]
        ax7.plot(mejora_acumulada, color='darkgreen', linewidth=2)
        ax7.fill_between(range(len(mejora_acumulada)), mejora_acumulada, 
                        alpha=0.3, color='green')
        ax7.set_xlabel('Iteración', fontsize=11)
        ax7.set_ylabel('Mejora Acumulada', fontsize=11)
        ax7.set_title('Progreso de Optimización', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 8. Velocidad de convergencia
        ax8 = plt.subplot(3, 3, 8)
        ventana_derivada = 50
        derivadas = []
        for i in range(ventana_derivada, len(self.historial_mejor_energia)):
            derivada = abs(self.historial_mejor_energia[i] - 
                          self.historial_mejor_energia[i-ventana_derivada])
            derivadas.append(derivada)
        
        ax8.plot(derivadas, color='brown', linewidth=1.5)
        ax8.set_xlabel('Iteración', fontsize=11)
        ax8.set_ylabel('Cambio en Mejor Energía', fontsize=11)
        ax8.set_title('Velocidad de Convergencia', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.set_yscale('log')
        
        # 9. Estadísticas finales (incluyendo métricas de calidad)
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        stats_text = f"""
        ESTADÍSTICAS FINALES
        {'='*35}
        
        Iteraciones totales: {len(self.historial_energia)}
        Saltos aleatorios: {self.contador_saltos}
        
        Mejor energía: {-self.historial_mejor_energia[-1]:.6f}
        Energía inicial: {-self.historial_energia[0]:.6f}
        Mejora total: {(-self.historial_mejor_energia[-1] - (-self.historial_energia[0])):.6f}
        
        Temperatura inicial: {self.historial_temperatura[0]:.2f}
        Temperatura final: {self.historial_temperatura[-1]:.4f}
        
        Tasa de saltos: {(self.contador_saltos/len(self.historial_energia)*100):.2f}%
        """
        
        # Agregar métricas de calidad si están disponibles
        if self.metricas_calidad:
            stats_text += f"""
        
        MÉTRICAS DE CALIDAD
        {'='*35}
        MSE: {self.metricas_calidad['MSE']:.4f}
        PSNR: {self.metricas_calidad['PSNR']:.2f} dB
        SSIM: {self.metricas_calidad['SSIM']:.4f}
        """
        
        ax9.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
    
    def visualizar_resultados(self, umbrales, imagen_segmentada):
        """
        Visualiza los resultados de la segmentación
        """
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Imagen original
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2RGB))
        ax1.set_title('Imagen Original', fontsize=13, fontweight='bold')
        ax1.axis('off')
        
        # 2. Imagen en escala de grises
        ax2 = plt.subplot(2, 4, 2)
        ax2.imshow(self.imagen_gris, cmap='gray')
        ax2.set_title('Escala de Grises', fontsize=13, fontweight='bold')
        ax2.axis('off')
        
        # 3. Imagen segmentada (jet)
        ax3 = plt.subplot(2, 4, 3)
        im3 = ax3.imshow(imagen_segmentada, cmap='jet')
        ax3.set_title(f'Segmentación ({len(umbrales)+1} niveles)', 
                     fontsize=13, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. Diferencia absoluta (para visualizar error)
        ax4 = plt.subplot(2, 4, 4)
        diferencia = np.abs(self.imagen_gris.astype(float) - imagen_segmentada.astype(float))
        im4 = ax4.imshow(diferencia, cmap='hot')
        ax4.set_title('Diferencia Absoluta (Error)', fontsize=13, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        # 5. Histograma con umbrales
        ax5 = plt.subplot(2, 4, 5)
        ax5.plot(self.histograma * 1000, color='black', linewidth=1.5)
        ax5.fill_between(range(256), self.histograma * 1000, alpha=0.3, color='gray')
        
        colores = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
        for i, umbral in enumerate(umbrales):
            color = colores[i % len(colores)]
            ax5.axvline(x=umbral, color=color, linestyle='--', linewidth=2.5,
                       label=f'Umbral {i+1}: {int(umbral)}')
        
        ax5.set_xlabel('Intensidad de Píxel', fontsize=11)
        ax5.set_ylabel('Frecuencia Normalizada (×1000)', fontsize=11)
        ax5.set_title('Histograma con Umbrales Óptimos', fontsize=13, fontweight='bold')
        ax5.legend(loc='upper right', fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # 6. Segmentación (viridis)
        ax6 = plt.subplot(2, 4, 6)
        im6 = ax6.imshow(imagen_segmentada, cmap='viridis')
        ax6.set_title('Segmentación (Viridis)', fontsize=13, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        
        # 7. Segmentación (plasma)
        ax7 = plt.subplot(2, 4, 7)
        im7 = ax7.imshow(imagen_segmentada, cmap='plasma')
        ax7.set_title('Segmentación (Plasma)', fontsize=13, fontweight='bold')
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
        
        # 8. Métricas de calidad (texto)
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')
        
        if self.metricas_calidad:
            metricas_text = f"""
            MÉTRICAS DE CALIDAD
            {'='*40}
            
            MSE (Error Cuadrático Medio)
            Valor: {self.metricas_calidad['MSE']:.4f}
            Menor es mejor (0 = idéntico)
            
            {'─'*40}
            
            PSNR (Relación Señal-Ruido)
            Valor: {self.metricas_calidad['PSNR']:.2f} dB
            """
            
            if self.metricas_calidad['PSNR'] > 40:
                metricas_text += "Calidad: Excelente "
            elif self.metricas_calidad['PSNR'] > 30:
                metricas_text += "Calidad: Buena "
            elif self.metricas_calidad['PSNR'] > 20:
                metricas_text += "Calidad: Aceptable"
            else:
                metricas_text += "Calidad: Baja "
            
            metricas_text += f"""
            
            {'─'*40}
            
            SSIM (Similitud Estructural)
            Valor: {self.metricas_calidad['SSIM']:.4f}
            """
            
            if self.metricas_calidad['SSIM'] > 0.9:
                metricas_text += "Calidad: Excelente "
            elif self.metricas_calidad['SSIM'] > 0.8:
                metricas_text += "Calidad: Buena "
            elif self.metricas_calidad['SSIM'] > 0.7:
                metricas_text += "Calidad: Aceptable "
            else:
                metricas_text += "Calidad: Baja "
            
            ax8.text(0.1, 0.5, metricas_text, fontsize=10, family='monospace',
                    verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
    
    def visualizar_metricas_graficas(self):
        """
        Visualiza las métricas de calidad en gráficas de barras
        """
        if not self.metricas_calidad:
            print("  No hay métricas calculadas. Ejecuta calcular_todas_metricas() primero.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Gráfica MSE
        axes[0].bar(['MSE'], [self.metricas_calidad['MSE']], color='red', alpha=0.7)
        axes[0].set_ylabel('Valor', fontsize=12)
        axes[0].set_title('MSE (Menor es mejor)', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].text(0, self.metricas_calidad['MSE'], 
                    f"{self.metricas_calidad['MSE']:.2f}", 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Gráfica PSNR
        color_psnr = 'green' if self.metricas_calidad['PSNR'] > 30 else 'orange'
        axes[1].bar(['PSNR'], [self.metricas_calidad['PSNR']], color=color_psnr, alpha=0.7)
        axes[1].set_ylabel('dB', fontsize=12)
        axes[1].set_title('PSNR (Mayor es mejor)', fontsize=13, fontweight='bold')
        axes[1].axhline(y=30, color='red', linestyle='--', linewidth=2, label='Umbral bueno (30 dB)')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].legend()
        axes[1].text(0, self.metricas_calidad['PSNR'], 
                    f"{self.metricas_calidad['PSNR']:.2f} dB", 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Gráfica SSIM
        color_ssim = 'green' if self.metricas_calidad['SSIM'] > 0.8 else 'orange'
        axes[2].bar(['SSIM'], [self.metricas_calidad['SSIM']], color=color_ssim, alpha=0.7)
        axes[2].set_ylabel('Valor', fontsize=12)
        axes[2].set_ylim([0, 1])
        axes[2].set_title('SSIM (Mayor es mejor, máx=1)', fontsize=13, fontweight='bold')
        axes[2].axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Umbral bueno (0.8)')
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].legend()
        axes[2].text(0, self.metricas_calidad['SSIM'], 
                    f"{self.metricas_calidad['SSIM']:.4f}", 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def guardar_resultados(self, imagen_segmentada, nombre_base='segmentada_SA'):
        """
        Guarda la imagen segmentada
        """
        cv2.imwrite(f'{nombre_base}.jpg', imagen_segmentada)
        print(f"✓ Imagen segmentada guardada como '{nombre_base}.jpg'")


def ejecutar_temple_simulado_completo(ruta_imagen, num_umbrales=3):
    """
    Función principal para ejecutar el Temple Simulado con Saltos Aleatorios
    
    Args:
        ruta_imagen: Ruta de la imagen
        num_umbrales: Número de umbrales (2-5 recomendado)
    """
    try:
        # Crear objeto de Temple Simulado
        ts = TempleSimuladoConSaltos(ruta_imagen, num_umbrales)
        
        # Ejecutar optimización con Temple Simulado
        umbrales_optimos = ts.optimizar(
            temperatura_inicial=100,        # Temperatura inicial alta para exploración
            temperatura_final=0.01,         # Temperatura final baja
            alpha=0.95,                     # Factor de enfriamiento (0.9-0.99)
            iteraciones_por_temp=50,        # Iteraciones en cada temperatura
            prob_salto=0.05,                # 5% probabilidad de salto aleatorio
            iteraciones_sin_mejora_max=200, # Salto forzado tras 200 iter sin mejora
            esquema_enfriamiento='exponencial'  # 'exponencial' o 'logaritmico'
        )
        
        # Segmentar imagen
        imagen_segmentada = ts.segmentar_imagen(umbrales_optimos)
        
        # ===== CALCULAR MÉTRICAS DE CALIDAD =====
        print("\n🔍 Calculando métricas de calidad...")
        metricas = ts.calcular_todas_metricas(imagen_segmentada)
        ts.mostrar_metricas(metricas)
        
        # Visualizar convergencia
        ts.visualizar_convergencia()
        
        # Visualizar resultados de segmentación (ahora incluye métricas)
        ts.visualizar_resultados(umbrales_optimos, imagen_segmentada)
        
        # Visualizar métricas en gráficas
        ts.visualizar_metricas_graficas()
        
        # Guardar resultados
        ts.guardar_resultados(imagen_segmentada, 
                             f'segmentada_SA_{num_umbrales}niveles')
        
        return ts, umbrales_optimos, imagen_segmentada
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# EJEMPLO DE USO
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" TEMPLE SIMULADO CON SALTOS ALEATORIOS")
    print(" Segmentación Multinivel de Imágenes con Métricas de Calidad")
    print("="*70 + "\n")
    
    # ========== CONFIGURA AQUÍ ==========
    ruta_imagen = 'example.jpg'  # <-- CAMBIA ESTO por tu imagen
    num_umbrales = 3                # Número de umbrales (2-5 recomendado)
    # ====================================
    
    # Ejecutar Temple Simulado
    ts, umbrales, img_segmentada = ejecutar_temple_simulado_completo(
        ruta_imagen, 
        num_umbrales
    )
    
    if ts is not None:
        print("\n" + "="*70)
        print(" PROCESO COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"\n Umbrales encontrados: {[int(u) for u in umbrales]}")
        print(f" Saltos aleatorios realizados: {ts.contador_saltos}")
        print(f"\n MÉTRICAS DE CALIDAD:")
        print(f"   • MSE:  {ts.metricas_calidad['MSE']:.4f}")
        print(f"   • PSNR: {ts.metricas_calidad['PSNR']:.2f} dB")
        print(f"   • SSIM: {ts.metricas_calidad['SSIM']:.4f}")
        print("="*70)
