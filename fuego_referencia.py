from modelo_rdc import spread_infection_raw
import numpy as np 
import cupy as cp # type: ignore

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

# Parámetros del modelo SI
beta_veg = cp.where(vegetacion <= 2, 0, 0.1 * vegetacion) # fracción de vegetación incéndiandose por hora

# Hacemos una máscara donde vegetación <=2, gamma >> 1/dt. Sino, vale 0.1. 
gamma = cp.where(vegetacion <= 2, cp.float32(100), cp.float32(0.1)) # fracción de vegetación que se apaga por hora.
                                                                    # 1/gamma es el tiempo promedio del incendio

beta_veg = beta_veg.astype(cp.float32)
gamma = gamma.astype(cp.float32)

dt = cp.float32(1/6) # Paso temporal. Si medimos el tiempo en horas, 1/6 indica un paso de 10 minutos

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
B = cp.float32(15) # m/h

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

# Coordenadas del punto de ignición
x_ignicion = 500
y_ignicion = 550

if vegetacion[y_ignicion, x_ignicion] > 2:
    # Infectados en una esquina de la grilla
    S[y_ignicion, x_ignicion] = 0
    I[y_ignicion, x_ignicion] = 1

    var_poblacion = 0

    # Inicializar arrays de cupy para almacenar los resultados
    num_steps = 10001
    # pob_total = cp.zeros(num_steps)
    # S_total = cp.zeros(num_steps)
    # I_total = cp.zeros(num_steps)
    # R_total = cp.zeros(num_steps)

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()

    # Definir arrays de estado
    S_new = cp.empty_like(S)
    I_new = cp.empty_like(I)
    R_new = cp.empty_like(R)

    S, I, R, S_new, I_new, R_new, beta_veg, gamma, wx, wy, h_dx_mapa, h_dy_mapa = ensure_batch_dim(
    S, I, R, S_new, I_new, R_new, beta_veg, gamma, wx, wy, h_dx_mapa, h_dy_mapa)

    # Iterar sobre las simulaciones
    for t in range(num_steps):
        spread_infection_raw(S, I, R, S_new, I_new, R_new, dt, d, beta_veg, gamma, D, wx, wy, h_dx_mapa, h_dy_mapa, A, B)

        # Swap de buffers (intercambiar referencias en lugar de crear nuevos arrays)
        S, S_new = S_new, S
        I, I_new = I_new, I
        R, R_new = R_new, R

        # suma_S = S.sum() / (nx*ny)
        # suma_I = I.sum() / (nx*ny)
        # suma_R = R.sum() / (nx*ny)

        # suma_total = suma_S + suma_I + suma_R
        # pob_total[t] = suma_total
        # S_total[t] = suma_S
        # I_total[t] = suma_I
        # R_total[t] = suma_R

        # var_poblacion += cp.abs(suma_total - pob_total[t-1]) if t > 0 else 0

    end.record()  # Marca el final en GPU
    end.synchronize() # Sincroniza y mide el tiempo

    # var_poblacion_promedio = var_poblacion / num_steps

    # print(f'Variación de población promedio: {var_poblacion_promedio}')

    # Calcular el número de celdas quemadas
    celdas_quemadas = cp.sum(R > 0.2)
    print(f'Número de celdas quemadas: {celdas_quemadas}')

    # Guardar el estado final de R en un archivo
    np.save("R_final.npy", cp.asnumpy(R))

    gpu_time = cp.cuda.get_elapsed_time(start, end)  # Tiempo en milisegundos
    print(f"Tiempo en GPU: {gpu_time:.3f} ms")
else:
    print("El punto de ignición corresponde a una celda no combustible.")