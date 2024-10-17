from PIL import Image, ImageOps, ImageFilter
import numpy as np

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

def image_to_binary_pattern(pattern_base, threshold=128, grid_size=(10, 10)):
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

# Ejemplo de uso
pattern_base = ["./imagen1.jpg", "./imagen2.jpg", "./imagen3.jpg"]  # Lista de imágenes
threshold = 128  # Ajusta el umbral aquí
image_to_binary_pattern(pattern_base, threshold)

# Para verificar el resultado, puedes imprimir una pequeña parte de la matriz
#np.set_printoptions(threshold=np.inf)


