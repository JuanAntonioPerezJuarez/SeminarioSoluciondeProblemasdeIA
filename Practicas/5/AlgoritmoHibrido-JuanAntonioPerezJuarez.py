import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List

class HybridGDPSO:
    """
    Algoritmo de Optimización Híbrido: Gradiente Descendente + PSO
    
    Combina la exploración global de PSO con la explotación local de GD
    """
    
    def __init__(self, 
                 objective_function: Callable,
                 gradient_function: Callable,
                 bounds: List[Tuple[float, float]],
                 n_particles: int = 30,
                 max_iterations: int = 100,
                 w: float = 0.7,           # Inercia PSO
                 c1: float = 1.5,          # Coeficiente cognitivo PSO
                 c2: float = 1.5,          # Coeficiente social PSO
                 learning_rate: float = 0.01,  # Tasa de aprendizaje GD
                 gd_iterations: int = 5,   # Iteraciones GD por ciclo
                 switch_threshold: float = 0.01):  # Umbral para cambiar de estrategia
        
        self.f = objective_function
        self.grad_f = gradient_function
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.dim = len(bounds)
        
        # Parámetros PSO
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Parámetros GD
        self.learning_rate = learning_rate
        self.gd_iterations = gd_iterations
        self.switch_threshold = switch_threshold
        
        # Inicialización
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.history = []
        
    def initialize_swarm(self):
        """Inicializa el enjambre de partículas"""
        self.particles = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            (self.n_particles, self.dim)
        )
        
        # Velocidades iniciales
        velocity_range = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
        self.velocities = np.random.uniform(
            -velocity_range, 
            velocity_range, 
            (self.n_particles, self.dim)
        )
        
        # Inicializar mejores personales
        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = np.array([self.f(p) for p in self.particles])
        
        # Inicializar mejor global
        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[best_idx].copy()
        self.global_best_score = self.personal_best_scores[best_idx]
        
    def pso_step(self):
        """Ejecuta un paso de PSO"""
        r1 = np.random.random((self.n_particles, self.dim))
        r2 = np.random.random((self.n_particles, self.dim))
        
        # Actualizar velocidades
        cognitive = self.c1 * r1 * (self.personal_best_positions - self.particles)
        social = self.c2 * r2 * (self.global_best_position - self.particles)
        self.velocities = self.w * self.velocities + cognitive + social
        
        # Actualizar posiciones
        self.particles += self.velocities
        
        # Aplicar límites
        self.particles = np.clip(self.particles, self.bounds[:, 0], self.bounds[:, 1])
        
        # Evaluar y actualizar mejores
        for i in range(self.n_particles):
            score = self.f(self.particles[i])
            
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.particles[i].copy()
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i].copy()
    
    def gradient_descent_step(self, position: np.ndarray) -> np.ndarray:
        """Ejecuta varias iteraciones de Gradiente Descendente desde una posición"""
        current_pos = position.copy()
        
        for _ in range(self.gd_iterations):
            gradient = self.grad_f(current_pos)
            new_pos = current_pos - self.learning_rate * gradient
            
            # Aplicar límites
            new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])
            
            # Si no hay mejora significativa, detener
            if np.linalg.norm(new_pos - current_pos) < 1e-8:
                break
                
            current_pos = new_pos
        
        return current_pos
    
    def hybrid_step(self, iteration: int):
        """
        Estrategia híbrida adaptativa:
        - Fase inicial: Más PSO (exploración)
        - Fase final: Más GD (explotación)
        """
        # Ejecutar paso PSO
        self.pso_step()
        
        # Determinar si aplicar GD basado en el progreso
        progress = iteration / self.max_iterations
        
        # Aplicar GD con mayor probabilidad en fases avanzadas
        if progress > 0.3:  # Después del 30% de iteraciones
            # Aplicar GD a las mejores partículas
            n_elite = max(1, int(self.n_particles * 0.2))  # Top 20%
            elite_indices = np.argsort(self.personal_best_scores)[:n_elite]
            
            for idx in elite_indices:
                # Aplicar GD desde la mejor posición personal
                refined_pos = self.gradient_descent_step(self.personal_best_positions[idx])
                refined_score = self.f(refined_pos)
                
                # Actualizar si hay mejora
                if refined_score < self.personal_best_scores[idx]:
                    self.personal_best_scores[idx] = refined_score
                    self.personal_best_positions[idx] = refined_pos
                    self.particles[idx] = refined_pos  # Mover la partícula
                    
                    if refined_score < self.global_best_score:
                        self.global_best_score = refined_score
                        self.global_best_position = refined_pos.copy()
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float, List]:
        """
        Ejecuta el algoritmo de optimización híbrido
        
        Returns:
            best_position: Mejor posición encontrada
            best_score: Mejor valor de la función objetivo
            history: Historia de convergencia
        """
        self.initialize_swarm()
        self.history = [self.global_best_score]
        
        if verbose:
            print(f"Iteración 0: Mejor valor = {self.global_best_score:.6f}")
        
        for iteration in range(1, self.max_iterations + 1):
            self.hybrid_step(iteration)
            self.history.append(self.global_best_score)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteración {iteration}: Mejor valor = {self.global_best_score:.6f}")
        
        # Refinamiento final con GD intensivo
        if verbose:
            print("\n--- Refinamiento final con Gradiente Descendente ---")
        
        final_position = self.global_best_position.copy()
        for _ in range(20):  # Más iteraciones GD al final
            final_position = self.gradient_descent_step(final_position)
        
        final_score = self.f(final_position)
        
        if final_score < self.global_best_score:
            self.global_best_score = final_score
            self.global_best_position = final_position
        
        if verbose:
            print(f"Valor final después de refinamiento: {self.global_best_score:.6f}")
        
        return self.global_best_position, self.global_best_score, self.history
    
    def plot_convergence(self):
        """Visualiza la convergencia del algoritmo"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history, linewidth=2)
        plt.xlabel('Iteración', fontsize=12)
        plt.ylabel('Mejor Valor de la Función Objetivo', fontsize=12)
        plt.title('Convergencia del Algoritmo Híbrido GD-PSO', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()


# ============= EJEMPLOS DE USO =============

def example_1_sphere():
    """Ejemplo 1: Función Esfera (convexa, simple)"""
    print("="*60)
    print("EJEMPLO 1: Función Esfera")
    print("="*60)
    
    # Función objetivo: sum(x_i^2)
    def sphere(x):
        return np.sum(x**2)
    
    # Gradiente: 2*x
    def sphere_grad(x):
        return 2 * x
    
    # Optimización
    bounds = [(-10, 10)] * 5  # 5 dimensiones
    optimizer = HybridGDPSO(
        objective_function=sphere,
        gradient_function=sphere_grad,
        bounds=bounds,
        n_particles=20,
        max_iterations=50,
        learning_rate=0.1
    )
    
    best_pos, best_score, history = optimizer.optimize()
    
    print(f"\n✓ Mejor posición: {best_pos}")
    print(f"✓ Mejor valor: {best_score:.10f}")
    print(f"✓ Óptimo teórico: 0.0")
    
    optimizer.plot_convergence()


def example_2_rosenbrock():
    """Ejemplo 2: Función Rosenbrock (no convexa, valle estrecho)"""
    print("\n" + "="*60)
    print("EJEMPLO 2: Función Rosenbrock")
    print("="*60)
    
    # Función objetivo: sum(100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2)
    def rosenbrock(x):
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    # Gradiente
    def rosenbrock_grad(x):
        grad = np.zeros_like(x)
        grad[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
        grad[1:] += 200 * (x[1:] - x[:-1]**2)
        return grad
    
    # Optimización
    bounds = [(-5, 5)] * 4  # 4 dimensiones
    optimizer = HybridGDPSO(
        objective_function=rosenbrock,
        gradient_function=rosenbrock_grad,
        bounds=bounds,
        n_particles=40,
        max_iterations=100,
        learning_rate=0.001,
        w=0.6,
        c1=2.0,
        c2=2.0
    )
    
    best_pos, best_score, history = optimizer.optimize()
    
    print(f"\n✓ Mejor posición: {best_pos}")
    print(f"✓ Mejor valor: {best_score:.10f}")
    print(f"✓ Óptimo teórico: [1, 1, 1, 1] con valor 0.0")
    
    optimizer.plot_convergence()


def example_3_rastrigin():
    """Ejemplo 3: Función Rastrigin (muchos mínimos locales)"""
    print("\n" + "="*60)
    print("EJEMPLO 3: Función Rastrigin")
    print("="*60)
    
    # Función objetivo
    def rastrigin(x):
        n = len(x)
        return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    # Gradiente
    def rastrigin_grad(x):
        return 2*x + 20*np.pi*np.sin(2*np.pi*x)
    
    # Optimización
    bounds = [(-5.12, 5.12)] * 3  # 3 dimensiones
    optimizer = HybridGDPSO(
        objective_function=rastrigin,
        gradient_function=rastrigin_grad,
        bounds=bounds,
        n_particles=50,
        max_iterations=150,
        learning_rate=0.01,
        w=0.7,
        c1=1.5,
        c2=1.5
    )
    
    best_pos, best_score, history = optimizer.optimize()
    
    print(f"\n✓ Mejor posición: {best_pos}")
    print(f"✓ Mejor valor: {best_score:.10f}")
    print(f"✓ Óptimo teórico: [0, 0, 0] con valor 0.0")
    
    optimizer.plot_convergence()


# Ejecutar ejemplos
if __name__ == "__main__":
    example_1_sphere()
    example_2_rosenbrock()
    example_3_rastrigin()
