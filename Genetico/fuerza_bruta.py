import cupy as cp # type: ignore
import os
from lectura_datos import preprocesar_datos, guardar_resultados
from operadores_geneticos import poblacion_inicial
from algoritmo import procesar_poblacion_batch
from config import cota, d, dt
import argparse
from pathlib import Path

print(cp.cuda.runtime.getDeviceProperties(0)['name'])

# Obtengo la variable por línea de comando
parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=int, default=1, help="Número de experimento")
args = parser.parse_args()

exp = args.exp

############################## CARGADO DE MAPAS #######################################################

datos = preprocesar_datos()
wx = datos["wx"]
wy = datos["wy"]
h_dx_mapa = datos["h_dx"]
h_dy_mapa = datos["h_dy"]

############################## INCENDIO DE REFERENCIA ####################################################

# directorio del archivo actual (Genetico/)
BASE_DIR = Path(__file__).resolve().parent

# directorio padre (raíz del proyecto)
PROJECT_DIR = BASE_DIR.parent

rutas = {
    1: PROJECT_DIR / "R_referencia_1.npy",
    2: PROJECT_DIR / "R_referencia_2.npy",
    3: PROJECT_DIR / "R_referencia_3.npy",
}

# Selecciono la ruta según EXP
ruta_incendio_referencia = rutas[exp]
print(f"Leyendo mapa de incendio de referencia: {os.path.basename(ruta_incendio_referencia)}")

############################## CONDICIÓN DE COURANT PARA LOS TÉRMINOS DIFUSIVOS Y ADVECTIVOS ####################################

A_max = float(d / (cp.sqrt(2)*dt/2*cp.max(cp.sqrt(wx**2+wy**2)))) # constante de viento
B_max = float(d / (cp.sqrt(2)*dt/2*cp.max(cp.sqrt(h_dx_mapa**2+h_dy_mapa**2)))) # constante de pendiente

############################## DISEÑO DE EXPERIMENTOS ##########################################

limite_parametros_base = [
    (0.01, 100.0),          # D
    (0.0, A_max * cota),    # A
    (0.0, B_max * cota)     # B
]

print(f"Corriendo el experimento {exp}")

if exp == 1:
    ajustar_beta_gamma = False
    ajustar_ignicion = True

    limite_ignicion = [(300, 720), (400, 800)]
    limite_parametros = limite_parametros_base + limite_ignicion

    beta_fijo = [0.91, 0.72, 1.38, 1.94, 0.75]
    gamma_fijo = [0.5, 0.38, 0.84, 0.45, 0.14]

    ignicion_fija_x = None
    ignicion_fija_y = None 

elif exp == 2:
    ajustar_beta_gamma = True
    ajustar_ignicion = True

    limite_ignicion = [(300, 720), (400, 800)]
    limite_beta = [(0.1, 2.0)]
    limite_gamma = [(0.1, 0.9)]
    limite_parametros = limite_parametros_base + limite_ignicion + limite_beta + limite_gamma

    beta_fijo = None
    gamma_fijo = None

    ignicion_fija_x = None
    ignicion_fija_y = None

elif exp == 3:
    ajustar_beta_gamma = True
    ajustar_ignicion = False

    limite_beta = [(0.1, 2.0)] * 5
    limite_gamma = [(0.1, 0.9)] * 5
    limite_parametros = limite_parametros_base + limite_beta + limite_gamma

    ignicion_fija_x = [1130, 1300, 620]
    ignicion_fija_y = [290, 150, 280]

    beta_fijo = None
    gamma_fijo = None

else:
    raise ValueError(f"Experimento {exp} no está definido")

############################## PARÁMETROS DE LOS INCENDIOS SIMULADOS ###############################################

# Total de simulaciones y tamaño de bloque
num_total_simulaciones = 100000
tamano_bloque = 10000
batch_size = 2

combinaciones = poblacion_inicial(num_total_simulaciones, limite_parametros) 

############################## BARRIDA DE PARÁMETROS ###############################################

start = cp.cuda.Event()
end = cp.cuda.Event()

start.record()

num_steps = 500
        
resultados_dir = f'resultados/fuerza_bruta_task_{exp}'
os.makedirs(resultados_dir, exist_ok=True)

# Loop por bloques
for i in range(0, num_total_simulaciones, tamano_bloque):
    # Determinar inicio y fin del bloque
    inicio = i
    fin = min(i + tamano_bloque, num_total_simulaciones)
    
    # Subconjunto de combinaciones para este bloque
    sub_poblacion = combinaciones[inicio:fin]

    print(f"Procesando simulaciones {inicio} a {fin}")

    # Ejecutar las simulaciones para esta parte
    resultados = procesar_poblacion_batch(
        poblacion=sub_poblacion,
        ruta_incendio_referencia=ruta_incendio_referencia,
        limite_parametros=limite_parametros,
        num_steps=num_steps,
        batch_size=batch_size,
        ajustar_beta_gamma=ajustar_beta_gamma,
        beta_fijo=beta_fijo,
        gamma_fijo=gamma_fijo,
        ajustar_ignicion=ajustar_ignicion,
        ignicion_fija_x=ignicion_fija_x,
        ignicion_fija_y=ignicion_fija_y
    )

    # Guardar resultados de este bloque
    bloque_id = inicio // tamano_bloque

    num_archivo = i + bloque_id

    guardar_resultados(resultados, resultados_dir, num_archivo, 
                       ajustar_beta_gamma=ajustar_beta_gamma,
                       ajustar_ignicion=ajustar_ignicion)

    print(f'Resultados guardados en: {resultados_dir}')

end.record()  # Marca el final en GPU
end.synchronize() # Sincroniza y mide el tiempo

gpu_time = cp.cuda.get_elapsed_time(start, end)  # Tiempo en milisegundos
print(f"Tiempo en GPU: {gpu_time:.3f} ms")

