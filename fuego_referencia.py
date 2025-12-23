from modelo_rdc import spread_infection_adi, courant_batch
import numpy as np 
import cupy as cp # type: ignore
import cupyx.scipy.ndimage #type: ignore
from mapas.io_mapas import preprocesar_datos
import argparse

# Obtengo la variable por línea de comando
parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=int, default=1, help="Número de experimento")
parser.add_argument("--visualizar_mapas", action='store_true', help="Visualizar los mapas al final de la simulación")
args = parser.parse_args()

exp = args.exp
visualizar_mapas = args.visualizar_mapas

############################## FUNCIÓN PARA AGREGAR UNA DIMENSIÓN ##################    #############################

def create_batch(array_base, n_batch):
    # Se repite array_base n_batch veces en un bloque contiguo
    return cp.tile(array_base[cp.newaxis, :, :], (n_batch, 1, 1)).copy()

datos = preprocesar_datos()
vegetacion = datos["vegetacion"]
wx = datos["wx"]
wy = datos["wy"]
h_dx_mapa = datos["h_dx"]
h_dy_mapa = datos["h_dy"]

# Obtener dimensiones del mapa
ny, nx = vegetacion.shape  # Usamos cualquier mapa para obtener las dimensiones

############################## PARÁMETROS DEL INCENDIO DE REFERENCIA ###############################################

# Tamaño de cada celda
d = cp.float32(30) # metros
# Paso temporal
dt = cp.float32(1/2) # horas
# Coeficiente de difusión
D_value = cp.float32(10) # metros^2 / hora
# Constante A adimensional de viento
A_value = cp.float32(1e-4) 
# Constante B de pendiente
B_value = cp.float32(15) # m/h

# Tasas de ignición y extinción según experimento
if exp == 1 or exp == 3:
    beta_params = [0.91, 0.72, 1.38, 1.94, 0.75]
    gamma_params = [0.5, 0.38, 0.84, 0.45, 0.14]
elif exp == 2:
    beta_params = [1.5, 1.5, 1.5, 1.5, 1.5]
    gamma_params = [0.5, 0.5, 0.5, 0.5, 0.5]

# Crear mapas de beta y gamma según tipo de vegetación
veg_types = cp.array([3, 4, 5, 6, 7], dtype=cp.int32)
beta_veg = cp.zeros_like(vegetacion, dtype=cp.float32)
gamma = cp.zeros_like(vegetacion, dtype=cp.float32)
# Asignar beta_veg según el tipo de vegetación
for j, veg_type in enumerate(veg_types):
    mask = (vegetacion == veg_type)
    beta_veg = cp.where(mask, beta_params[j], beta_veg)
    gamma = cp.where(mask, gamma_params[j], gamma)

# Poner beta y gamma en cero para vegetación 0, 1 y 2
mask_no_veg = (vegetacion == 0) | (vegetacion == 1) | (vegetacion == 2)
beta_veg = cp.where(mask_no_veg, 0.0, beta_veg)
gamma = cp.where(mask_no_veg, 0.0, gamma)

print(f'Valores de beta: {beta_params}')
print(f'Valores de gamma: {gamma_params}')

# Suavizar los mapas de beta y gamma
beta_veg = cupyx.scipy.ndimage.gaussian_filter(beta_veg, sigma=10.0)
gamma = cupyx.scipy.ndimage.gaussian_filter(gamma, sigma=10.0)

n_batch = 1

D = cp.full((n_batch), D_value, dtype=cp.float32)
A = cp.full((n_batch), A_value, dtype=cp.float32)
B = cp.full((n_batch), B_value, dtype=cp.float32)

############################## INCENDIO DE REFERENCIA ###############################################

# Población inicial de susceptibles e infectados
S_batch = cp.ones((n_batch, ny, nx), dtype=cp.float32)
I_batch = cp.zeros_like(S_batch)
R_batch = cp.zeros_like(S_batch)

S_batch = cp.where(vegetacion <= 2, 0, S_batch)  # Celdas no vegetadas no son susceptibles

print(f'Se cumple la condición de Courant para el término advectivo: {courant_batch(dt/2, A, B, d, wx, wy, h_dx_mapa, h_dy_mapa)}')

# Coordenadas del punto de ignición según experimento
if exp == 1 or exp == 2:
    x_ignicion = cp.array([400])
    y_ignicion = cp.array([600])
elif exp == 3:
    x_ignicion = cp.array([1130,1300,620])
    y_ignicion = cp.array([290,150,280])

S_batch[:, y_ignicion, x_ignicion] = 0
I_batch[:, y_ignicion, x_ignicion] = 1

# Inicializar arrays de cupy para almacenar los resultados
num_steps = 500

# Definir arrays de estado
S_new_batch = cp.empty_like(S_batch)
I_new_batch = cp.empty_like(I_batch)
R_new_batch = cp.empty_like(R_batch)

beta_veg_batch = create_batch(beta_veg, n_batch)
gamma_batch = create_batch(gamma, n_batch)

start = cp.cuda.Event()
end = cp.cuda.Event()

start.record()

# Iterar sobre las simulaciones
for t in range(num_steps):
    spread_infection_adi(S_batch, I_batch, R_batch, S_new_batch, I_new_batch, R_new_batch, dt, d, beta_veg_batch, gamma_batch, D, wx, wy, h_dx_mapa, h_dy_mapa, A, B, vegetacion)

    # Swap de buffers (intercambiar referencias en lugar de crear nuevos arrays)
    S_batch, S_new_batch = S_new_batch, S_batch
    I_batch, I_new_batch = I_new_batch, I_batch
    R_batch, R_new_batch = R_new_batch, R_batch

end.record()  # Marca el final en GPU
end.synchronize() # Sincroniza y mide el tiempo

cp.save(f"R_referencia_{exp}.npy", R_new_batch)

gpu_time = cp.cuda.get_elapsed_time(start, end)  # Tiempo en milisegundos
print(f"Tiempo en GPU: {gpu_time:.3f} ms")

print(f'Numero de celdas quemadas: {cp.sum(R_new_batch > 0.001)}')

############################## VISUALIZACIÓN DE LOS MAPAS ###############################################

if visualizar_mapas:
    import matplotlib.pyplot as plt
    import scienceplots
    
    plt.style.use(['science', 'ieee'])

    # Definir los nuevos colores para los valores del archivo (0 a 7)
    vegetation_colors = np.array([
        [255, 0, 255],      # 0: NODATA (magenta)
        [199, 209, 207],    # 1: Sin combustible (gris claro)
        [0, 0, 255],        # 2: Lagos (azul)
        [0, 117, 0],        # 3: Bosque A (verde oscuro)
        [50, 200, 10],      # 4: Bosque B (verde brillante)
        [150, 0, 150],      # 5: Bosque I (morado)
        [122, 127, 50],     # 6: Pastizal (verde oliva)
        [0, 196, 83]        # 7: Arbustal (verde intenso)
    ]) / 255.0  # Escalar los valores RGB al rango [0, 1]

    # Mapear los valores de vegetación a colores RGB
    vegetation = vegetation_colors[vegetacion.get().astype(int)]

    # Crear visualización de vectores de viento sobre el mapa de vegetación
    fig, ax = plt.subplots()

    # Transferencia del mapa a CPU
    R_cpu = np.squeeze(R_new_batch.get())

    # Mostrar el mapa de vegetación como fondo
    terrain_rgb = (1 - np.clip(R_cpu[..., None], 0, 1)) * vegetation + np.clip(R_cpu[..., None], 0, 1) * np.array([1.0, 0.0, 0.0])
    im = ax.imshow(terrain_rgb, interpolation='nearest', origin='lower')

    x_ticks = np.arange(0, nx, 200)  # Cada 200 celdas en el eje X
    y_ticks = np.arange(0, ny, 200)  # Cada 200 celdas en el eje Y
    x_labels = (x_ticks * d) / 1000  # Convertir a kilómetros
    y_labels = (y_ticks * d) / 1000

    # Configurar ejes y etiquetas
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{x:.1f}" for x in x_labels])
    ax.set_yticklabels([f"{y:.1f}" for y in y_labels])
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")

    if exp == 1 or exp == 2:
        ax.scatter([400], [600], color='red', marker='*', s=10, edgecolors='black', linewidths=0.5)
    elif exp == 3:
        ax.scatter([1130,1300,620], [290,150,280], color='red', marker='*', s=10, edgecolors='black', linewidths=0.5)

    plt.tight_layout()
    plt.savefig(f'R_referencia_{exp}_map.pdf', transparent=True, dpi=600, bbox_inches='tight')
    plt.show()