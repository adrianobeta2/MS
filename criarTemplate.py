import cv2
import os

def create_template(image_path, output_path, x, y, w, h):
    """
    Recorta uma região de uma imagem (baseado em x, y, largura e altura) e salva como um novo arquivo.

    :param image_path: Caminho da imagem original.
    :param output_path: Caminho onde o template será salvo.
    :param x: Coordenada x do canto superior esquerdo do retângulo.
    :param y: Coordenada y do canto superior esquerdo do retângulo.
    :param w: Largura da região de recorte.
    :param h: Altura da região de recorte.
    """
    # Verificar se a imagem existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    
    # Carregar a imagem original
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
    
    # Verificar as dimensões da imagem
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height or w <= 0 or h <= 0:
        raise ValueError(f"Parâmetros de recorte inválidos. Tamanho da imagem: ({width}, {height})")
    
    # Recortar a região definida
    cropped_region = image[y:y+h, x:x+w]
    
    # Verificar se a região recortada é válida
    if cropped_region.size == 0:
        raise ValueError("A região recortada está vazia. Verifique as coordenadas fornecidas.")
    
    # Salvar a região recortada como arquivo
    if not cv2.imwrite(output_path, cropped_region):
        raise IOError(f"Erro ao salvar o template em: {output_path}")
    
    print(f"Template salvo com sucesso em: {output_path}")


# Exemplo de uso:
input_image = "captura.png"  # Caminho da imagem original
output_template = "template.png"  # Caminho para salvar o template
x, y, w, h = 169, 224, 7, 24  # Coordenadas (x, y) e dimensões (w, h)

try:
    create_template(input_image, output_template, x, y, w, h)
except Exception as e:
    print(f"Erro: {e}")

