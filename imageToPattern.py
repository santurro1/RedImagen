from PIL import Image, ImageOps, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Obtiene la ruta del archivo actual
base_dir = os.path.dirname(__file__)

# Directorio donde se guardarán las imágenes recortadas
cropped_dir = os.path.join(base_dir, "cropped")
os.makedirs(cropped_dir, exist_ok=True)  # Crea el directorio si no existe

# Clase para la red de Hopfield
class HopfieldNetwork:
    def __init__(self, num_neurons):
        """
        Inicializa la red de Hopfield.

        Parámetros:
        - num_neurons: Número de neuronas en la red (equivalente al tamaño del patrón).
        """
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))  # Matriz de pesos inicializada a cero

    def train(self, patterns):
        """
        Entrena la red con una lista de patrones.

        Parámetros:
        - patterns: Lista de patrones binarios para entrenar la red.
        """
        for p in patterns:
            self.weights += np.outer(p, p)  # Calcula el producto externo y lo suma a los pesos
        np.fill_diagonal(self.weights, 0)  # Establece los valores de la diagonal en 0
        self.weights /= len(patterns)  # Normaliza los pesos dividiendo por el número de patrones

    def predict(self, input_pattern, max_iterations=200):
        """
        Realiza la predicción usando el patrón de entrada, actualizando las neuronas en orden aleatorio.

        Parámetros:
        - input_pattern: Patrón de entrada que se va a recuperar.
        - max_iterations: Número máximo de iteraciones para la predicción.

        Retorna:
        - output_pattern: El patrón de salida después de la convergencia.
        """
        output_pattern = input_pattern.copy()
        for _ in range(max_iterations):
            indices = np.random.permutation(self.num_neurons)  # Actualización en orden aleatorio
            for i in indices:
                net_input = np.dot(self.weights[i], output_pattern)
                output_pattern[i] = 1 if net_input >= 0 else -1
        return output_pattern

def convert_alpha_to_white(image):
    """
    Convierte una imagen con transparencia en una imagen con fondo blanco.

    Parámetros:
    - image: Imagen en formato RGBA.

    Retorna:
    - Imagen en formato RGB con fondo blanco.
    """
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
    return image

def get_trimmed_image(image_path):
    """
    Recorta la imagen eliminando el espacio en blanco alrededor del objeto principal.

    Parámetros:
    - image_path: Ruta de la imagen.

    Retorna:
    - Imagen recortada en escala de grises.
    """
    original_image = Image.open(image_path)
    image = convert_alpha_to_white(original_image)
    image = image.convert("1")
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = ImageOps.invert(image)
    bbox = image.getbbox()
    cropped = convert_alpha_to_white(Image.open(image_path)).convert("L").crop(bbox)
    return cropped

def image_to_binary_pattern(pattern_base, threshold=128, grid_size=(20, 20)):
    """
    Convierte una lista de imágenes en patrones binarios.

    Parámetros:
    - pattern_base: Lista de rutas de las imágenes.
    - threshold: Umbral para convertir los valores de la imagen a binarios.
    - grid_size: Tamaño de la cuadrícula para redimensionar la imagen.

    Retorna:
    - Lista de patrones binarios en formato numpy array.
    """
    num = 1
    patterns_base = []
    Cantidad_patterns = len(pattern_base)
    for i in range(Cantidad_patterns):
        name_image = os.path.join(cropped_dir, f"cropped{num}.jpg")
        image = get_trimmed_image(pattern_base[i])
        image.save(name_image)
        image = image.resize(grid_size)
        image_array = np.array(image)

        # Genera la matriz binaria basada en el umbral
        binary_pattern = np.where(image_array > threshold, 0, 1)
        binary_pattern[binary_pattern == 0] = -1  # Reemplaza 0s con -1s para la red de Hopfield
        
        patterns_base.append(binary_pattern)
        name_txt = "pattern" + str(num) + ".txt"
        num += 1
        np.savetxt(name_txt, binary_pattern, fmt="%d")  # Guarda el patrón en un archivo txt
        
    return np.array(patterns_base)

def generate_noisy_patterns(base_pattern, num_patterns, noise_level):
    """
    Genera múltiples patrones ruidosos basados en un patrón original.

    Parámetros:
    - base_pattern: Patrón base sin ruido.
    - num_patterns: Número de patrones ruidosos a generar.
    - noise_level: Nivel de ruido (probabilidad de cambiar un valor en el patrón).

    Retorna:
    - Lista de patrones ruidosos.
    """
    noisy_patterns = []
    Cantidad_patterns = len(base_pattern)
    for i in range(Cantidad_patterns):
        for _ in range(num_patterns):
            noisy_pattern = base_pattern[i].copy()
            noise = np.random.choice([0, 1], size=base_pattern[i].shape, p=[1 - noise_level, noise_level])
            noisy_pattern = np.where(noise == 1, -noisy_pattern, noisy_pattern)
            noisy_patterns.append(noisy_pattern)
    return np.array(noisy_patterns)

def generated_random_pattern(base_pattern, noise_level):
    """
    Genera un patrón aleatorio similar a un patrón base añadiendo ruido.

    Parámetros:
    - base_pattern: Patrón base.
    - noise_level: Nivel de ruido (probabilidad de cambiar un valor en el patrón).

    Retorna:
    - Patrón aleatorio con ruido.
    """
    noisy_pattern = base_pattern.copy()
    noise = np.random.choice([0, 1], size=base_pattern.shape, p=[1 - noise_level, noise_level])
    noisy_pattern = np.where(noise == 1, -noisy_pattern, noisy_pattern)
    return noisy_pattern.copy()

# Configuración de la red de Hopfield y generación de patrones
num_neurons = 400  # Número de neuronas en la red (equivalente al tamaño del patrón)
hopfield_net = HopfieldNetwork(num_neurons)

# Lista de imágenes para los patrones base
pattern_base = [os.path.join(base_dir, "imagen1.jpg"), 
                os.path.join(base_dir, "imagen2.jpg"), 
                os.path.join(base_dir, "imagen3.jpg")]

threshold = 128
base_pattern = image_to_binary_pattern(pattern_base, threshold)

# Generación de patrones ruidosos para entrenamiento
noisy_patterns = generate_noisy_patterns(base_pattern, num_patterns=50, noise_level=0.1)

# Entrenamiento de la red
hopfield_net.train(noisy_patterns)

# Bucle para probar la red de Hopfield con diferentes patrones
start = 's'
while start == 's':
    tamano = len(noisy_patterns)
    num_rand = random.randint(0, tamano)

    print("Cantidad de patrones base:", len(pattern_base))
    print("Patron random: ", num_rand)
    print("Tamaño: ", tamano)
    base_pattern = noisy_patterns[num_rand].copy()

    # Genera un patrón de prueba con ruido
    test_pattern = generated_random_pattern(base_pattern, noise_level=0.1)
    test_pattern = test_pattern.flatten()

    # Mostrar el patrón inicial
    plt.subplot(1, 2, 1)
    plt.title("Patrón inicial")
    plt.imshow(test_pattern.reshape((20, 20)), cmap="gray")

    # Recupera el patrón usando la red de Hopfield
    output_pattern = hopfield_net.predict(test_pattern, max_iterations=200)

    # Mostrar el patrón recuperado
    plt.subplot(1, 2, 2)
    plt.title("Patrón recuperado")
    plt.imshow(output_pattern.reshape((20, 20)), cmap="gray")
    plt.show()

    # Solicita al usuario si desea continuar
    print("Desea seguir probando? (s-n)")
    start = input("Respuesta: ")



