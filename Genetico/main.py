import cupy as cp # type: ignore
import time
from config import d, dt, cota
from lectura_datos import preprocesar_datos
from algoritmo import genetic_algorithm

print(cp.cuda.runtime.getDeviceProperties(0)['name'])
print(cp.__version__)
print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
print("CUDA driver :", cp.cuda.runtime.driverGetVersion())

############################## CARGADO DE MAPAS #######################################################

datos = preprocesar_datos()
wx = datos["wx"]
wy = datos["wy"]
h_dx_mapa = datos["h_dx"]
h_dy_mapa = datos["h_dy"]

############################## INCENDIO DE REFERENCIA ####################################################

# ruta_incendio_referencia = 'c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin/area_quemada_SM.asc'
ruta_incendio_referencia = 'c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/R_referencia.npy'

############################## CONDICIÓN DE COURANT PARA LOS TÉRMINOS DIFUSIVOS Y ADVECTIVOS ############

A_max = float(d / (cp.sqrt(2)*dt/2*cp.max(cp.sqrt(wx**2+wy**2)))) # constante de viento
B_max = float(d / (cp.sqrt(2)*dt/2*cp.max(cp.sqrt(h_dx_mapa**2+h_dy_mapa**2)))) # constante de pendiente

############################## EJECUCIÓN DEL ALGORITMO ###############################################

limite_beta = [(0.1, 2.0)] * 5  # Límites para beta_veg
limite_gamma = [(0.1, 0.9)] * 5  # Límites para gamma

# Población aleatoria inicial (D, A, B, x, y)
limite_parametros = [(0.01, 100.), (0.0, A_max * cota), (0.0, B_max * cota), (300, 720), (400, 800)] + limite_beta + limite_gamma

# Sincronizar antes de empezar a medir el tiempo
cp.cuda.Stream.null.synchronize()
start_time = time.time()

# Ejecutar el GA con procesamiento en batch
resultados = genetic_algorithm(tamano_poblacion=20, generaciones=2, limite_parametros=limite_parametros,
                               ruta_incendio_referencia=ruta_incendio_referencia, 
                            #    archivo_preentrenado='resultados/task_default/resultados_generacion_1.csv', 
                               num_steps=500,
                               batch_size=5)

# Sincronizar después de completar la ejecución
cp.cuda.Stream.null.synchronize()
end_time = time.time()

print(f"Tiempo de ejecución en GPU: {end_time - start_time} segundos")

