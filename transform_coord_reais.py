import numpy as np

# Dados de calibração (exemplo)
# Pontos no espaço real (X, Y) em mm
real_points = np.array([
    [10, 20],
    [30, 40],
    [50, 60],
    [70, 80]
])

# Pontos correspondentes na imagem (x, y) em pixels
pixel_points = np.array([
    [100, 150],
    [200, 250],
    [300, 350],
    [400, 450]
])

# Adicionar uma coluna de 1s para a translação
A = np.hstack([pixel_points, np.ones((pixel_points.shape[0], 1))])

# Resolver o sistema para X e Y separadamente
X_params, _, _, _ = np.linalg.lstsq(A, real_points[:, 0], rcond=None)
Y_params, _, _, _ = np.linalg.lstsq(A, real_points[:, 1], rcond=None)

# Matriz de transformação (2x3)
transform_matrix = np.vstack([X_params, Y_params])

print("Matriz de transformação:")
print(transform_matrix)

# Função para converter coordenadas de pixel para reais
def pixel_to_real(x, y):
    pixel_vector = np.array([x, y, 1])  # Vetor de pixel (3x1)
    real_vector = np.dot(transform_matrix, pixel_vector)  # Multiplicação (2x3) * (3x1) = (2x1)
    return real_vector

# Teste de conversão
x_pixel, y_pixel = 250, 300  # Exemplo de coordenada de pixel
X_real, Y_real = pixel_to_real(x_pixel, y_pixel)
print(f"Coordenadas reais: X = {X_real:.2f} mm, Y = {Y_real:.2f} mm")