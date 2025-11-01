from modelo_rdc import spread_infection_adi, courant_batch
import numpy as np 
import cupy as cp # type: ignore
import cupyx.scipy.ndimage #type: ignore
from Genetico.config import ruta_mapas

############################## FUNCIÓN PARA AGREGAR UNA DIMENSIÓN ##################    #############################

def create_batch(array_base, n_batch):
    # Se repite array_base n_batch veces en un bloque contiguo
    return cp.tile(array_base[cp.newaxis, :, :], (n_batch, 1, 1)).copy()

############################## LEER MAPAS RASTER ###############################################

def leer_asc(ruta):
    with open(ruta, 'r') as f:
        for _ in range(6):  
            f.readline()
        data = np.loadtxt(f) 
        return cp.array(data, dtype=cp.float32) 

############################## CALCULO A PARTIR DE MAPAS ###############################################

def preprocesar_datos():
    datos = [leer_asc(m) for m in ruta_mapas]
    vientod, vientov, pendiente, vegetacion, orientacion = datos

    # Parámetros derivados
    beta_veg = cp.where(vegetacion <= 2, 0, 0.1 * vegetacion)
    gamma = cp.where(vegetacion <= 2, 100, 0.1)

    vientod_rad = vientod * cp.pi / 180
    # Componentes cartesianas del viento:
    wx = -vientov * cp.sin(vientod_rad) * 1000  # Este = sin(ángulo desde Norte)
    wy = -vientov * cp.cos(vientod_rad) * 1000 # Norte = cos(ángulo desde Norte)

    slope = pendiente / 100.0 # pendiente en porcentaje
    orientacion_rad = orientacion * cp.pi / 180

    h_dx = -slope * cp.sin(orientacion_rad)
    h_dy = -slope * cp.cos(orientacion_rad)

    return {
        "vientod": cp.flipud(vientod),
        "vientov": cp.flipud(vientov),
        "pendiente": cp.flipud(pendiente),
        "vegetacion": cp.flipud(vegetacion),
        "orientacion": cp.flipud(orientacion),
        "beta_veg": cp.flipud(beta_veg),
        "gamma": cp.flipud(gamma),
        "wx": cp.flipud(wx),
        "wy": cp.flipud(wy),
        "h_dx": cp.flipud(h_dx),
        "h_dy": cp.flipud(h_dy),
        "ny": vientod.shape[0],
        "nx": vientod.shape[1],
    }

############################## CARGADO DE MAPAS ###############################################

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
# Coeficiente de difusión
D_value = cp.float32(10) # metros^2 / hora. Si la celda tiene 30 metros, en una hora avanza 1/3 del tamaño de la celda

# beta_params = [1.5, 1.5, 1.5, 1.5, 1.5]
# gamma_params = [0.5, 0.5, 0.5, 0.5, 0.5]

beta_params = [0.91, 0.72, 1.38, 1.94, 0.75]
gamma_params = [0.5, 0.38, 0.84, 0.45, 0.14]

# beta_params = [0.724, 0.537, 0.970, 1.976, 0.893]
# gamma_params = [0.297, 0.185, 0.442, 0.428, 0.316]

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

beta_veg = beta_veg.astype(cp.float32)
gamma = gamma.astype(cp.float32)

# Guardado de mapas
np.save("beta_veg.npy", beta_veg.get())
np.save("gamma.npy", gamma.get())

beta_veg = cupyx.scipy.ndimage.gaussian_filter(beta_veg, sigma=10.0)
gamma = cupyx.scipy.ndimage.gaussian_filter(gamma, sigma=10.0)

dt = cp.float32(1/2) # Paso temporal

# Constante A adimensional de viento
A_value = cp.float32(1e-4) # 10^-3 está al doble del límite de estabilidad

# Constante B de pendiente
B_value = cp.float32(15) # m/h

n_batch = 1

D = cp.full((n_batch), D_value, dtype=cp.float32)
A = cp.full((n_batch), A_value, dtype=cp.float32)
B = cp.full((n_batch), B_value, dtype=cp.float32)

############################## INCENDIO DE REFERENCIA ###############################################

# Población inicial de susceptibles e infectados
S_batch = cp.ones((n_batch, ny, nx), dtype=cp.float32)
I_batch = cp.zeros_like(S_batch)
R_batch = cp.zeros_like(S_batch)

S_batch = cp.where(vegetacion <= 2, 0, S_batch)  # Celdas no vegetadas son susceptibles

print(f'Se cumple la condición de Courant para el término advectivo: {courant_batch(dt/2, A, B, d, wx, wy, h_dx_mapa, h_dy_mapa)}')

# Coordenadas del punto de ignición
x_ignicion = cp.array([1130, 1300, 620])
y_ignicion = cp.array([290, 150, 280])

S_batch[:, y_ignicion, x_ignicion] = 0
I_batch[:, y_ignicion, x_ignicion] = 1

# var_poblacion = 0

# Inicializar arrays de cupy para almacenar los resultados
num_steps = 500
# pob_total = cp.zeros(num_steps)
# S_total = cp.zeros(num_steps)
# I_total = cp.zeros(num_steps)
# R_total = cp.zeros(num_steps)

# Definir arrays de estado
S_new_batch = cp.empty_like(S_batch)
I_new_batch = cp.empty_like(I_batch)
R_new_batch = cp.empty_like(R_batch)

# celdas_rotas = cp.zeros(S.shape[0], dtype=cp.int32)

beta_veg_batch = create_batch(beta_veg, n_batch)
gamma_batch = create_batch(gamma, n_batch)

# print(f'El array es contiguo: {vegetacion.flags.c_contiguous}')

# Sumas por batch
# suma_S = S.sum(axis=(1,2)) / (nx*ny)  # array de tamaño n_batch
# suma_I = I.sum(axis=(1,2)) / (nx*ny)
# suma_R = R.sum(axis=(1,2)) / (nx*ny)
# suma_total = suma_S + suma_I + suma_R

# for b in range(S.shape[0]):
#     print(f"batch {b}: Total = {suma_total[b]:.3f}, S = {suma_S[b]:.3f}, I = {suma_I[b]:.3f}, R = {suma_R[b]:.3f}")
#     print(f'  S: min={S[b].min():.3f}, max={S[b].max():.3f}')
#     print(f'  I: min={I[b].min():.3f}, max={I[b].max():.3f}')
#     print(f'  R: min={R[b].min():.3f}, max={R[b].max():.3f}')

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

    # # Revisar por batch
    # for b in range(S.shape[0]):
    #     if not cp.all((S[b] <= 1) & (S[b] >= 0)):
    #         celdas_rotas[b] += cp.sum((S[b] < 0) | (S[b] > 1))
    #     if not cp.all((I[b] <= 1) & (I[b] >= 0)):
    #         celdas_rotas[b] += cp.sum((I[b] < 0) | (I[b] > 1))
    #     if not cp.all((R[b] <= 1) & (R[b] >= 0)):
    #         celdas_rotas[b] += cp.sum((R[b] < 0) | (R[b] > 1))

    # # Sumas por batch
    # suma_S = S.sum(axis=(1,2)) / (nx*ny)  # array de tamaño n_batch
    # suma_I = I.sum(axis=(1,2)) / (nx*ny)
    # suma_R = R.sum(axis=(1,2)) / (nx*ny)
    # suma_total = suma_S + suma_I + suma_R

    # pob_total[t] = suma_total.mean()  # promedio sobre batches
    # S_total[t] = suma_S.mean()
    # I_total[t] = suma_I.mean()
    # R_total[t] = suma_R.mean()

    # if (t % 100 == 0) or (t == num_steps - 1):
    #     for b in range(S.shape[0]):
    #       print(f"Paso {t}, batch {b}: Total = {suma_total[b]:.3f}, S = {suma_S[b]:.3f}, I = {suma_I[b]:.3f}, R = {suma_R[b]:.3f}")
    #       print(f'  S: min={S[b].min():.3f}, max={S[b].max():.3f}')
    #       print(f'  I: min={I[b].min():.3f}, max={I[b].max():.3f}')
    #       print(f'  R: min={R[b].min():.3f}, max={R[b].max():.3f}')

    # var_poblacion += cp.abs(pob_total[t] - pob_total[t-1]) if t > 0 else 0

end.record()  # Marca el final en GPU
end.synchronize() # Sincroniza y mide el tiempo

cp.save("R_referencia_3.npy", R_new_batch)

gpu_time = cp.cuda.get_elapsed_time(start, end)  # Tiempo en milisegundos
print(f"Tiempo en GPU: {gpu_time:.3f} ms")

# var_poblacion_promedio = var_poblacion / num_steps

# print(f'Variación de población promedio: {var_poblacion_promedio}')
# print(f'Número de celdas rotas: {celdas_rotas}')
print(f'Numero de celdas quemadas: {cp.sum(R_new_batch > 0.001)}')


