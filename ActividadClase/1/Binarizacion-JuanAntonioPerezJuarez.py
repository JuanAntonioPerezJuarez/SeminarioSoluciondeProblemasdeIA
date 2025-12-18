import cv2
import matplotlib.pyplot as plt

def binarizar_imagen_simple(ruta_imagen, umbral=127):
    """
    Binariza una imagen usando el método simple con umbral fijo
    
    Args:
        ruta_imagen: Ruta de la imagen a binarizar
        umbral: Valor del umbral (0-255), por defecto 127
    
    Returns:
        Imagen binarizada
    """
    # Leer la imagen
    imagen = cv2.imread(ruta_imagen)
    
    if imagen is None:
        print("Error: No se pudo cargar la imagen")
        return None
    
    # Convertir a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar binarización simple
    _, imagen_binaria = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    
    return imagen, imagen_gris, imagen_binaria


def mostrar_resultados(imagen_original, imagen_gris, imagen_binaria):
    """
    Muestra la imagen original, en gris y binarizada
    """
    plt.figure(figsize=(15, 5))
    
    # Imagen original
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original')
    plt.axis('off')
    
    # Imagen en escala de grises
    plt.subplot(1, 3, 2)
    plt.imshow(imagen_gris, cmap='gray')
    plt.title('Escala de Grises')
    plt.axis('off')
    
    # Imagen binarizada
    plt.subplot(1, 3, 3)
    plt.imshow(imagen_binaria, cmap='gray')
    plt.title('Imagen Binarizada')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# USO DEL CÓDIGO
if __name__ == "__main__":
    # Cambia esta ruta por la ruta de tu imagen
    ruta_imagen = '../ActividadClase/example.jpg'  # <-- CAMBIA ESTO
    
    # Cambia el umbral si lo deseas (valores entre 0 y 255)
    umbral = 127
    
    # Binarizar la imagen
    original, gris, binaria = binarizar_imagen_simple(ruta_imagen, umbral)
    
    if binaria is not None:
        # Mostrar resultados
        mostrar_resultados(original, gris, binaria)
        
        # Guardar la imagen binarizada
        cv2.imwrite('imagen_binarizada.jpg', binaria)
        print(f"✓ Imagen binarizada guardada como 'imagen_binarizada.jpg'")
        print(f"✓ Umbral utilizado: {umbral}")
