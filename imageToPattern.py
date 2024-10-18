from PIL import Image, ImageOps, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import random

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def predict(self, input_pattern, max_iterations=200):
        output_pattern = input_pattern.copy()
        for _ in range(max_iterations):
            indices = np.random.permutation(self.num_neurons)  # Actualización en orden aleatorio
            for i in indices:
                net_input = np.dot(self.weights[i], output_pattern)
                output_pattern[i] = 1 if net_input >= 0 else -1
        return output_pattern

def convert_alpha_to_white(image):
    if image.mode == "RGBA":
        # Crear una imagen de fondo blanca
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        # Combinar la imagen con el fondo blanco
        image = Image.alpha_composite(background, image).convert("RGB")
    return image

def get_trimmed_image(image_path):
    original_image = Image.open(image_path)
    image = convert_alpha_to_white(original_image)
    image = image.convert("1")
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = ImageOps.invert(image)
    bbox = image.getbbox()
    cropped = convert_alpha_to_white(Image.open(image_path)).convert("L").crop(bbox)
    return cropped

def image_to_binary_pattern(pattern_base, threshold=128, grid_size=(20, 20)):
    num = 1
    patterns_base = []
    for i in range(3):
        name_image = "./imagenes/cropped" + str(num) + ".jpg"
        image = get_trimmed_image(pattern_base[i])
        image.save(name_image)
        image = image.resize(grid_size)
        image_array = np.array(image)

        # Genera la matriz binaria basada en el umbral
        binary_pattern = np.where(image_array > threshold, 0, 1)

        # Reemplaza los 0 por -1
        binary_pattern[binary_pattern == 0] = -1
        
        patterns_base.append(binary_pattern)

        # Definimos título
        name_txt = "pattern" + str(num) + ".txt"
        num += 1

        # Guardar el patrón en un archivo txt
        np.savetxt(name_txt, binary_pattern, fmt="%d")
        
    return np.array(patterns_base)

# Generar 50 patrones de la letra A con ruido
def generate_noisy_patterns(base_pattern, num_patterns, noise_level):
    noisy_patterns = []
    for i in range(3):
        for _ in range(num_patterns):
            noisy_pattern = base_pattern[i].copy()
            noise = np.random.choice([0, 1], size=base_pattern[i].shape, p=[1 - noise_level, noise_level])
            noisy_pattern = np.where(noise == 1, -noisy_pattern, noisy_pattern)
            noisy_patterns.append(noisy_pattern)
    return np.array(noisy_patterns)

# Gene patron random similar a la letra A para probar
def generated_random_pattern(base_pattern, noise_level):
    noisy_pattern = base_pattern.copy()
    noise = np.random.choice([0, 1], size=base_pattern.shape, p=[1 - noise_level, noise_level])
    noisy_pattern = np.where(noise == 1, -noisy_pattern, noisy_pattern)
    patterns = noisy_pattern.copy()
    return np.array(patterns)

# Crear la red de Hopfield
num_neurons = 400
hopfield_net = HopfieldNetwork(num_neurons)

# Ejemplo de uso
pattern_base = ["./imagen1.jpg", "./imagen2.jpg", "./imagen3.jpg"]  # Lista de imágenes
threshold = 128  # Ajusta el umbral aquí
base_pattern = image_to_binary_pattern(pattern_base, threshold)

noisy_patterns = generate_noisy_patterns(base_pattern, num_patterns=50, noise_level=0.1) #forzar falla

# Entrenar la red con los patrones ruidosos
hopfield_net.train(noisy_patterns)

# Tomar uno de los patrones de entrenamiento con ruido
base_pattern = []
tamano = len(noisy_patterns)
num_rand = random.randint(0, tamano)

print("Patron random: ", num_rand)
print("Tamaño: ", tamano)
base_pattern = noisy_patterns[num_rand].copy()
# Agarramos un pater de prueba
test_pattern = generated_random_pattern(base_pattern, noise_level=0.1)
test_pattern = test_pattern.flatten()  # Convierte una matriz en un vector

# Mostrar el patrón inicial
plt.subplot(1, 2, 1)
plt.title("Patrón inicial")
plt.imshow(test_pattern.reshape((20, 20)), cmap="gray")

# Recuperar el patrón usando la red de Hopfield
output_pattern = hopfield_net.predict(test_pattern, max_iterations=200)

# Mostrar el patrón recuperado
plt.subplot(1, 2, 2)
plt.title("Patrón recuperado")
plt.imshow(output_pattern.reshape((20, 20)), cmap="gray")
plt.show()


