from modelo_rdc import spread_infection_adi, courant
import numpy as np 
import cupy as cp # type: ignore
import cupyx.scipy.ndimage

############################## FUNCIÓN PARA AGREGAR UNA DIMENSIÓN ###############################################

def ensure_batch_dim(*arrays):
    return [arr if arr.ndim == 3 else arr[cp.newaxis, ...] for arr in arrays]

############################## CARGADO DE MAPAS ###############################################

# Ruta de los archivos
ruta_mapas = ['c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin/ang_wind.asc',   # Dirección del viento
              'c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin/speed_wind.asc', # Velocidad del viento
              'c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin/asc_slope.asc',  # Pendiente del terreno
              'c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin/asc_CIEFAP.asc', # Vegetación
              'c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin/asc_aspect.asc', # Orientación del terreno
]

# Función para leer archivos .asc
def leer_asc(ruta):
    with open(ruta, 'r') as f:
        # Leer el encabezado para obtener el tamaño de la grilla
        for _ in range(6):  # Las primeras 6 líneas suelen contener los metadatos
            f.readline()
        
        # Leer el resto de los datos y convertirlos a una matriz
        data = np.loadtxt(f)  # Leer el archivo de datos en un array de numpy
        return cp.array(data, dtype=cp.float32)

# Leer los mapas
datos = [leer_asc(mapa) for mapa in ruta_mapas]

# Asignar cada matriz a una variable (e invertir verticalmente)
vientod = cp.flipud(datos[0])
vientov = cp.flipud(datos[1])
pendiente = cp.flipud(datos[2])
vegetacion = cp.flipud(datos[3])
orientacion = cp.flipud(datos[4])

# Obtener dimensiones del mapa
ny, nx = vegetacion.shape  # Usamos cualquier mapa para obtener las dimensiones

############################## PARÁMETROS DEL INCENDIO DE REFERENCIA ###############################################

# Tamaño de cada celda
d = cp.float32(30) # metros
# Coeficiente de difusión
D = cp.float32(10) # metros^2 / hora. Si la celda tiene 30 metros, en una hora avanza 1/3 del tamaño de la celda

# Sortear valores aleatorios para beta_params y gamma_params
rng = cp.random.default_rng(seed=45)
beta_params = rng.uniform(0.2, 2, size=5)
gamma_params = rng.uniform(0.05, 1, size=5)

# Asegurar que beta > gamma en cada posición
for i in range(5):
    if beta_params[i] <= gamma_params[i]:
        gamma_params[i] = beta_params[i] - rng.uniform(0.05, 0.15)
        if gamma_params[i] < 0.01:
            gamma_params[i] = 0.01

beta_params = beta_params.tolist()
gamma_params = gamma_params.tolist()

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

# Parámetros del modelo SI
# beta_veg = cp.where(vegetacion <= 2, cp.float32(0), 0.1 * vegetacion) # fracción de vegetación incéndiandose por hora

# Hacemos una máscara donde vegetación <=2, gamma >> 1/dt. Sino, vale 0.1. 
# gamma = cp.where(vegetacion <= 2, cp.float32(0), cp.float32(0.1)) # fracción de vegetación que se apaga por hora.
                                                                    # 1/gamma es el tiempo promedio del incendio

beta_veg = beta_veg.astype(cp.float32)
gamma = gamma.astype(cp.float32)

# Guardado de mapas
np.save("beta_veg.npy", beta_veg.get())
np.save("gamma.npy", gamma.get())

beta_veg = cupyx.scipy.ndimage.gaussian_filter(beta_veg, sigma=10.0)
gamma = cupyx.scipy.ndimage.gaussian_filter(gamma, sigma=10.0)

dt = cp.float32(2/3) # Paso temporal. Si medimos el tiempo en horas, 1/2 indica un paso de 30 minutos

# Transformación del viento a coordenadas cartesianas
# El viento está medido en km/h. En m/h el viento es una cantidad enorme, por eso
# la cantidad A que multiplica el viento tendría que ser pequeña. Intuición: A~10^-5

# Convertir a radianes
vientod_rad = vientod * cp.pi / 180
# Componentes cartesianas del viento:
# wx: componente Este-Oeste (positivo hacia el Este)
# wy: componente Norte-Sur (positivo hacia el Norte)
wx = -vientov * cp.sin(vientod_rad) * 1000  # Este = sin(ángulo desde Norte)
wy = -vientov * cp.cos(vientod_rad) * 1000  # Norte = cos(ángulo desde Norte)

# Constante A adimensional de viento
A = cp.float32(1e-4) # 10^-3 está al doble del límite de estabilidad

# Constante B de pendiente
B = cp.float32(1) # m/h

D = cp.asarray(D, dtype=cp.float32).reshape(1)
A = cp.asarray(A, dtype=cp.float32).reshape(1)
B = cp.asarray(B, dtype=cp.float32).reshape(1)

# Cálculo de la pendiente (usando mapas de pendiente y orientación)
h_dx_mapa = (cp.tan(pendiente * cp.pi / 180) * cp.cos(orientacion * cp.pi / 180 - cp.pi/2)).astype(cp.float32)
h_dy_mapa = (cp.tan(pendiente * cp.pi / 180) * cp.sin(orientacion * cp.pi / 180 - cp.pi/2)).astype(cp.float32)

############################## INCENDIO DE REFERENCIA ###############################################

# Población inicial de susceptibles e infectados
S = cp.ones((ny, nx), dtype=cp.float32)  # Todos son susceptibles inicialmente
I = cp.zeros((ny, nx), dtype=cp.float32) # Ningún infectado al principio
R = cp.zeros((ny, nx), dtype=cp.float32)

S = cp.where(vegetacion <= 2, 0, S)  # Celdas no vegetadas son susceptibles

print(f'Se cumple la condición de Courant para el término advectivo: {courant(dt/2, D, A, B, d, wx, wy, h_dx_mapa, h_dy_mapa)}')

# Coordenadas del punto de ignición
x_ignicion = 699
y_ignicion = 727

if vegetacion[y_ignicion, x_ignicion] > 2:
    # Infectados en una esquina de la grilla
    S[y_ignicion, x_ignicion] = 0
    I[y_ignicion, x_ignicion] = 1

    var_poblacion = 0

    # Inicializar arrays de cupy para almacenar los resultados
    num_steps = 2000
    pob_total = cp.zeros(num_steps)
    S_total = cp.zeros(num_steps)
    I_total = cp.zeros(num_steps)
    R_total = cp.zeros(num_steps)

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()

    # Definir arrays de estado
    S_new = cp.empty_like(S)
    I_new = cp.empty_like(I)
    R_new = cp.empty_like(R)

    celdas_rotas = 0

    S, I, R, S_new, I_new, R_new, beta_veg, gamma, wx, wy, h_dx_mapa, h_dy_mapa = ensure_batch_dim(
    S, I, R, S_new, I_new, R_new, beta_veg, gamma, wx, wy, h_dx_mapa, h_dy_mapa)

    # Iterar sobre las simulaciones
    for t in range(num_steps):
        spread_infection_adi(S, I, R, S_new, I_new, R_new, dt, d, beta_veg, gamma, D, wx, wy, h_dx_mapa, h_dy_mapa, A, B, vegetacion)

        # Swap de buffers (intercambiar referencias en lugar de crear nuevos arrays)
        S, S_new = S_new, S
        I, I_new = I_new, I
        R, R_new = R_new, R

        # if not cp.all((S <= 1) & (S >= 0)):
        #     # print(f"Error: Valores de S fuera de rango en el paso {t}")
        #     celdas_rotas += cp.sum((S < 0) | (S > 1))
        #     # break

        # if not cp.all((I <= 1) & (I >= 0)):
        #     # print(f"Error: Valores de I fuera de rango en el paso {t}")
        #     celdas_rotas += cp.sum((I < 0) | (I > 1))
        #     # print(f"Total de celdas rotas hasta el paso {t}: {celdas_rotas}")
        #     # print(f'Valor mínimo de I: {I.min()}')
        #     # print(f'Valor máximo de I: {I.max()}')
        #     # break

        # if not cp.all((R <= 1) & (R >= 0)):
        #     # print(f"Error: Valores de R fuera de rango en el paso {t}")
        #     celdas_rotas += cp.sum((R < 0) | (R > 1))
        #     # print(f"Total de celdas rotas hasta el paso {t}: {celdas_rotas}")
        #     # print(f'Valor mínimo de R: {R.min()}')
        #     # print(f'Valor máximo de R: {R.max()}')
        #     # break

        # suma_S = S.sum() / (nx*ny)
        # suma_I = I.sum() / (nx*ny)
        # suma_R = R.sum() / (nx*ny)

        # suma_total = suma_S + suma_I + suma_R
        # pob_total[t] = suma_total
        # S_total[t] = suma_S
        # I_total[t] = suma_I
        # R_total[t] = suma_R

        # if (t % 500 == 0) or (t == num_steps - 1):
        #     print(f"Paso {t}: Población total = {suma_total}, Susceptibles = {suma_S}, Infectados = {suma_I}, Recuperados = {suma_R}")
        #     print(f'Valor máximo de S: {S.max()}')
        #     print(f'Valor mínimo de S: {S.min()}')
        #     print(f'Valor máximo de I: {I.max()}')
        #     print(f'Valor mínimo de I: {I.min()}')
        #     print(f'Valor máximo de R: {R.max()}')
        #     print(f'Valor mínimo de R: {R.min()}')

        #     mask = (vegetacion == 1)
        #     print(f'Valor máximo de R en no combustibles: {R[mask].max()}')

        # var_poblacion += cp.abs(suma_total - pob_total[t-1]) if t > 0 else 0

    end.record()  # Marca el final en GPU
    end.synchronize() # Sincroniza y mide el tiempo

    np.save("R_final.npy", cp.asnumpy(R_new))

    gpu_time = cp.cuda.get_elapsed_time(start, end)  # Tiempo en milisegundos
    print(f"Tiempo en GPU: {gpu_time:.3f} ms")

    var_poblacion_promedio = var_poblacion / num_steps

    print(f'Variación de población promedio: {var_poblacion_promedio}')
    print(f'Número de celdas rotas: {celdas_rotas}')

else:
    print("El punto de ignición corresponde a una celda no combustible.")