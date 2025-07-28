from modelo_rdc import spread_infection_raw
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
D = cp.float32(21.625) # metros^2 / hora. Si la celda tiene 30 metros, en una hora avanza 1/3 del tamaño de la celda

# beta_params = [0.3, 0.4, 0.5, 0.6, 0.7]
# gamma_params = [0.1, 0.1, 0.1, 0.1, 0.1]

# veg_types = cp.array([3, 4, 5, 6, 7], dtype=cp.int32)
# beta_veg = cp.zeros_like(vegetacion, dtype=cp.float32)
# gamma = cp.zeros_like(vegetacion, dtype=cp.float32)
# # Asignar beta_veg según el tipo de vegetación
# for j, veg_type in enumerate(veg_types):
#     mask = (vegetacion == veg_type)
#     beta_veg = cp.where(mask, beta_params[j], beta_veg)
#     gamma = cp.where(mask, gamma_params[j], gamma)

#gamma = cupyx.scipy.ndimage.gaussian_filter(gamma, sigma=3.0)

# Parámetros del modelo SI
beta_veg = cp.where(vegetacion <= 2, cp.float32(0), cp.float32(0.4)) # fracción de vegetación incéndiandose por hora

# Hacemos una máscara donde vegetación <=2, gamma >> 1/dt. Sino, vale 0.1. 
gamma = cp.where(vegetacion <= 2, cp.float32(0), cp.float32(0.1)) # fracción de vegetación que se apaga por hora.
                                                                    # 1/gamma es el tiempo promedio del incendio

beta_veg = beta_veg.astype(cp.float32)
gamma = gamma.astype(cp.float32)

dt = cp.float32(1/4) # Paso temporal. Si medimos el tiempo en horas, 1/6 indica un paso de 10 minutos

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
A = cp.float32(7e-4) # 10^-3 está al doble del límite de estabilidad

# Constante B de pendiente
B = cp.float32(10.4521) # m/h

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

# Coordenadas del punto de ignición
x_ignicion = 699
y_ignicion = 727

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

    # ESTADÍSTICAS Y DIAGNÓSTICO
    print("ANÁLISIS DE PARÁMETROS INICIALES")
    print("=" * 50)
    
    # Análisis de beta y gamma
    beta_stats = {
        'min': float(cp.min(beta_veg)),
        'max': float(cp.max(beta_veg)),
        'mean': float(cp.mean(beta_veg)),
        'std': float(cp.std(beta_veg))
    }
    
    gamma_stats = {
        'min': float(cp.min(gamma)),
        'max': float(cp.max(gamma)),
        'mean': float(cp.mean(gamma)),
        'std': float(cp.std(gamma))
    }
    
    print(f"Beta - Min: {beta_stats['min']:.3f}, Max: {beta_stats['max']:.3f}, Mean: {beta_stats['mean']:.3f}, Std: {beta_stats['std']:.3f}")
    print(f"Gamma - Min: {gamma_stats['min']:.3f}, Max: {gamma_stats['max']:.3f}, Mean: {gamma_stats['mean']:.3f}, Std: {gamma_stats['std']:.3f}")
    
    # Análisis de condición de estabilidad
    beta_gamma_sum = beta_veg + gamma
    stability_condition = 1 / beta_gamma_sum
    dt_val = float(dt)
    
    print(f"\nCONDICIÓN DE ESTABILIDAD (dt < 1/(β+γ))")
    print(f"dt actual: {dt_val:.4f}")
    print(f"β+γ - Min: {float(cp.min(beta_gamma_sum)):.3f}, Max: {float(cp.max(beta_gamma_sum)):.3f}")
    print(f"dt_max requerido - Min: {float(cp.min(stability_condition)):.4f}, Max: {float(cp.max(stability_condition)):.4f}")
    
    # Verificar violaciones de estabilidad (quitar dimensión de batch para análisis)
    beta_gamma_2d = beta_gamma_sum[0]  # Quitar dimensión de batch
    violaciones = beta_gamma_2d > (1/dt_val)
    num_violaciones = int(cp.sum(violaciones))
    if num_violaciones > 0:
        print(f"⚠️ ADVERTENCIA: {num_violaciones} celdas violan la condición de estabilidad!")
        violacion_coords = cp.where(violaciones)
        for i in range(min(5, len(violacion_coords[0]))):  # Mostrar solo las primeras 5
            y, x = int(violacion_coords[0][i]), int(violacion_coords[1][i])
            bg_sum = float(beta_gamma_2d[y, x])
            print(f"  Celda ({y},{x}): β+γ={bg_sum:.3f}, dt_max_requerido={1/bg_sum:.4f}")
    else:
        print("✅ Todas las celdas cumplen la condición de estabilidad")
    
    # Análisis por tipo de vegetación (también usar arrays 2D)
    print(f"\nANÁLISIS POR TIPO DE VEGETACIÓN")
    veg_types = cp.unique(vegetacion)
    for veg_type in veg_types:
        mask = vegetacion == veg_type
        count = int(cp.sum(mask))
        if count > 0:
            beta_mean = float(cp.mean(beta_veg[0][mask]))  # Usar índice [0] para quitar batch dim
            gamma_mean = float(cp.mean(gamma[0][mask]))    # Usar índice [0] para quitar batch dim
            bg_sum_mean = beta_mean + gamma_mean
            dt_max = 1/bg_sum_mean if bg_sum_mean > 0 else float('inf')
            estable = "✅" if dt_val < dt_max else "❌"
            print(f"Veg {int(veg_type)}: {count:4d} celdas, β={beta_mean:.3f}, γ={gamma_mean:.3f}, dt_max={dt_max:.4f} {estable}")

    celdas_con_errores = 0
    errores_por_tipo = {'S': 0, 'I': 0, 'R': 0}
    pasos_con_errores = []

    # Iterar sobre las simulaciones
    for t in range(num_steps):
        spread_infection_raw(S, I, R, S_new, I_new, R_new, dt, d, beta_veg, gamma, D, wx, wy, h_dx_mapa, h_dy_mapa, A, B)

        # Swap de buffers (intercambiar referencias en lugar de crear nuevos arrays)
        S, S_new = S_new, S
        I, I_new = I_new, I
        R, R_new = R_new, R

        if cp.any((R > 1) | (R < 0)) or cp.any((S < 0) | (S > 1)) or cp.any((I < 0) | (I > 1)):
            pasos_con_errores.append(t)
            
            # Contar errores por tipo
            if cp.any((R > 1) | (R < 0)):
                errores_por_tipo['R'] += 1
                print(f"Error: Valores de R fuera de rango en el paso {t}")
                
                # Estadísticas detalladas de R
                R_flat = R.flatten()
                print(f"  R - Min: {float(cp.min(R_flat)):.6f}, Max: {float(cp.max(R_flat)):.6f}")
                print(f"  R - Valores < 0: {int(cp.sum(R < 0))}, Valores > 1: {int(cp.sum(R > 1))}")
                
                if cp.any(R > 1):
                    max_R = float(cp.max(R))
                    max_coords = cp.unravel_index(cp.argmax(R), R.shape)
                    print(f"  Valor máximo R: {max_R:.6f} en coordenada {max_coords}")
                if cp.any(R < 0):
                    min_R = float(cp.min(R))
                    min_coords = cp.unravel_index(cp.argmin(R), R.shape)
                    print(f"  Valor mínimo R: {min_R:.6f} en coordenada {min_coords}")
                    
            if cp.any((S < 0) | (S > 1)):
                errores_por_tipo['S'] += 1
                print(f"Error: Valores de S fuera de rango en el paso {t}")
                
                # Estadísticas detalladas de S
                S_flat = S.flatten()
                print(f"  S - Min: {float(cp.min(S_flat)):.6f}, Max: {float(cp.max(S_flat)):.6f}")
                print(f"  S - Valores < 0: {int(cp.sum(S < 0))}, Valores > 1: {int(cp.sum(S > 1))}")
                
                if cp.any(S > 1):
                    max_S = float(cp.max(S))
                    max_coords = cp.unravel_index(cp.argmax(S), S.shape)
                    print(f"  Valor máximo S: {max_S:.6f} en coordenada {max_coords}")
                if cp.any(S < 0):
                    min_S = float(cp.min(S))
                    min_coords = cp.unravel_index(cp.argmin(S), S.shape)
                    print(f"  Valor mínimo S: {min_S:.6f} en coordenada {min_coords}")
                    
            if cp.any((I < 0) | (I > 1)):
                errores_por_tipo['I'] += 1
                print(f"Error: Valores de I fuera de rango en el paso {t}")
                
                # Estadísticas detalladas de I
                I_flat = I.flatten()
                print(f"  I - Min: {float(cp.min(I_flat)):.6f}, Max: {float(cp.max(I_flat)):.6f}")
                print(f"  I - Valores < 0: {int(cp.sum(I < 0))}, Valores > 1: {int(cp.sum(I > 1))}")
                
                if cp.any(I > 1):
                    max_I = float(cp.max(I))
                    max_coords = cp.unravel_index(cp.argmax(I), I.shape)
                    print(f"  Valor máximo I: {max_I:.6f} en coordenada {max_coords}")
                    # Analizar parámetros en esa coordenada
                    y_max, x_max = max_coords[1], max_coords[2]  # Ajuste para batch dimension
                    beta_val = float(beta_veg[0, y_max, x_max])
                    gamma_val = float(gamma[0, y_max, x_max])
                    veg_val = float(vegetacion[y_max, x_max])
                    print(f"  En coordenada problema: β={beta_val:.6f}, γ={gamma_val:.6f}, veg={veg_val}")
                if cp.any(I < 0):
                    min_I = float(cp.min(I))
                    min_coords = cp.unravel_index(cp.argmin(I), I.shape)
                    print(f"  Valor mínimo I: {min_I:.6f} en coordenada {min_coords}")
            
            # Verificar conservación de masa
            total_SIR = S + I + R
            masa_min = float(cp.min(total_SIR))
            masa_max = float(cp.max(total_SIR))
            masa_mean = float(cp.mean(total_SIR))
            print(f"  Conservación masa S+I+R - Min: {masa_min:.6f}, Max: {masa_max:.6f}, Mean: {masa_mean:.6f}")
            
            celdas_con_errores += 1
            
            # # Si hay demasiados errores consecutivos, parar
            # if len(pasos_con_errores) >= 5 and all(p in pasos_con_errores for p in range(t-4, t+1)):
            #     print(f"Demasiados errores consecutivos. Parando simulación en paso {t}")
            #     break
            # break

        # Monitoreo periódico cada 500 pasos
        if t % 500 == 0 and t > 0:
            print(f"\nPaso {t} - Monitoreo periódico:")
            print(f"  S: min={float(cp.min(S)):.4f}, max={float(cp.max(S)):.4f}")
            print(f"  I: min={float(cp.min(I)):.4f}, max={float(cp.max(I)):.4f}")
            print(f"  R: min={float(cp.min(R)):.4f}, max={float(cp.max(R)):.4f}")
            
            # Masa total
            masa_total = float(cp.mean(S + I + R))
            print(f"  Masa promedio S+I+R: {masa_total:.6f}")
            
            # Progreso del incendio
            celdas_quemandose = int(cp.sum(I > 0.001))
            celdas_quemadas_parcial = int(cp.sum(R > 0.001))
            print(f"  Celdas quemándose: {celdas_quemandose}, Quemadas: {celdas_quemadas_parcial}")

        # if not cp.all((R <= 1) & (R >= 0)):
        #     print(f"Error: Valores de R fuera de rango en el paso {t}")
        #     break

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

    # ESTADÍSTICAS FINALES
    print("\n" + "=" * 50)
    print("ESTADÍSTICAS FINALES DE LA SIMULACIÓN")
    print("=" * 50)
    
    # Resumen de errores
    print(f"Errores detectados:")
    print(f"  Total pasos con errores: {len(pasos_con_errores)}")
    print(f"  Errores en S: {errores_por_tipo['S']}")
    print(f"  Errores en I: {errores_por_tipo['I']}")
    print(f"  Errores en R: {errores_por_tipo['R']}")
    
    if pasos_con_errores:
        print(f"  Primeros pasos con errores: {pasos_con_errores[:10]}")
        if len(pasos_con_errores) > 10:
            print(f"  ... y {len(pasos_con_errores)-10} más")
    
    # Estadísticas finales de valores
    print(f"\nValores finales:")
    print(f"  S - Min: {float(cp.min(S)):.6f}, Max: {float(cp.max(S)):.6f}, Mean: {float(cp.mean(S)):.6f}")
    print(f"  I - Min: {float(cp.min(I)):.6f}, Max: {float(cp.max(I)):.6f}, Mean: {float(cp.mean(I)):.6f}")
    print(f"  R - Min: {float(cp.min(R)):.6f}, Max: {float(cp.max(R)):.6f}, Mean: {float(cp.mean(R)):.6f}")
    
    # Conservación de masa final
    total_final = S + I + R
    print(f"  Conservación masa S+I+R - Min: {float(cp.min(total_final)):.6f}, Max: {float(cp.max(total_final)):.6f}")
    
    # Estadísticas de propagación del incendio
    celdas_afectadas = int(cp.sum((I > 0.001) | (R > 0.001)))
    celdas_quemandose = int(cp.sum(I > 0.001))
    celdas_quemadas_final = int(cp.sum(R > 0.001))
    
    print(f"\nPropagación del incendio:")
    print(f"  Celdas total afectadas: {celdas_afectadas}")
    print(f"  Celdas quemándose (I>0.001): {celdas_quemandose}")
    print(f"  Celdas quemadas (R>0.001): {celdas_quemadas_final}")

    # var_poblacion_promedio = var_poblacion / num_steps

    # print(f'Variación de población promedio: {var_poblacion_promedio}')

    # Calcular el número de celdas quemadas
    celdas_quemadas = cp.sum(R > 0.001)
    print(f'\nNúmero de celdas quemadas (threshold 0.001): {celdas_quemadas}')

    # Guardar el estado final de R en un archivo
    np.save("R_final.npy", cp.asnumpy(R))

    gpu_time = cp.cuda.get_elapsed_time(start, end)  # Tiempo en milisegundos
    print(f"Tiempo en GPU: {gpu_time:.3f} ms")
else:
    print("El punto de ignición corresponde a una celda no combustible.")