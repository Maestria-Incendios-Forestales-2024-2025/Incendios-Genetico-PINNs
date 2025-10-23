import cupy as cp # type: ignore
import time
from config import d, dt, cota
from lectura_datos import preprocesar_datos
from algoritmo import genetic_algorithm
import socket, os
import argparse

print(cp.cuda.runtime.getDeviceProperties(0)['name'])

cp.cuda.Device(0).use()

# Obtengo la variable por línea de comando
parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=int, default=1, help="Número de experimento")
parser.add_argument("--pretrained", type=str, default=None, help="Ruta al archivo preentrenado")
parser.add_argument("--start_gen", type=int, default=0, help="Generación desde la que entrenar")
args = parser.parse_args()

exp = args.exp

############################## CARGADO DE MAPAS #######################################################

datos = preprocesar_datos()
wx = datos["wx"]
wy = datos["wy"]
h_dx_mapa = datos["h_dx"]
h_dy_mapa = datos["h_dy"]

############################## INCENDIO DE REFERENCIA ####################################################

# Detecto dónde estoy corriendo
hostname = socket.gethostname()
# print(hostname)

# if "rocks7frontend" in hostname or "compute" in hostname:
#     base_cluster = "/home/lucas.becerra/Incendios-Genetico-PINNs/"
#     rutas = {
#         1: base_cluster + "R_referencia_1.npy",
#         2: base_cluster + "R_referencia_2.npy",
#         3: base_cluster + "R_referencia_3.npy",
#     }
# else:
#     base_local = "c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/"
#     rutas = {
#         1: base_local + "R_referencia_1.npy",
#         2: base_local + "R_referencia_2.npy",
#         3: base_local + "R_referencia_3.npy",
#     }

# Selecciono la ruta según EXP
# ruta_incendio_referencia = rutas[exp]
# print(f"Leyendo mapa de incendio de referencia: {os.path.basename(ruta_incendio_referencia)}")

if "ccad.unc.edu.ar" in hostname:
    ruta_incendio_referencia = "/home/lbecerra/Incendios-Genetico-PINNs/mapas_steffen_martin/area_quemada_SM.asc"
else: 
    ruta_incendio_referencia = "c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin/area_quemada_SM.asc"
    

############################## CARGA DE ARCHIVO PREENTRENADO ####################################

archivo_preentrenado = args.pretrained
generacion_preentranada = args.start_gen

############################## CONDICIÓN DE COURANT PARA LOS TÉRMINOS DIFUSIVOS Y ADVECTIVOS ####################################

A_max = float(d / (cp.sqrt(2)*dt/2*cp.max(cp.sqrt(wx**2+wy**2)))) # constante de viento
B_max = float(d / (cp.sqrt(2)*dt/2*cp.max(cp.sqrt(h_dx_mapa**2+h_dy_mapa**2)))) # constante de pendiente

############################## DISEÑO DE EXPERIMENTOS ##########################################

limite_parametros_base = [
    (0.01, 200.0),          # D
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

elif exp == 2:
    ajustar_beta_gamma = True
    ajustar_ignicion = True

    limite_ignicion = [(300, 720), (400, 800)]
    limite_beta = [(0.1, 2.0)]
    limite_gamma = [(0.1, 0.9)]
    limite_parametros = limite_parametros_base + limite_ignicion + limite_beta + limite_gamma

# Adapto el experimento 3 para ajustar el incendio real
elif exp == 3:
    ajustar_beta_gamma = True
    ajustar_ignicion = False

    limite_beta = [(0.1, 3.0)] * 4
    limite_gamma = [(0.1, 3.0)] * 4
    limite_parametros = limite_parametros_base + limite_beta + limite_gamma

    ignicion_fija_x = [475, 565]
    ignicion_fija_y = [550, 530]

else:
    raise ValueError(f"Experimento {exp} no está definido")

############################## EJECUCIÓN DEL ALGORITMO ###############################################

# Sincronizar antes de empezar a medir el tiempo
cp.cuda.Stream.null.synchronize()
start_time = time.time()

resultados = genetic_algorithm(
    tamano_poblacion=5000,
    generaciones=40,
    limite_parametros=limite_parametros,
    ruta_incendio_referencia=ruta_incendio_referencia,
    archivo_preentrenado=archivo_preentrenado,
    generacion_preentrenada=generacion_preentranada,
    num_steps=1920,
    batch_size=5,
    ajustar_beta_gamma=ajustar_beta_gamma,
    beta_fijo=beta_fijo if not ajustar_beta_gamma else None,
    gamma_fijo=gamma_fijo if not ajustar_beta_gamma else None,
    ajustar_ignicion=ajustar_ignicion,
    ignicion_fija_x=ignicion_fija_x if not ajustar_ignicion else None,
    ignicion_fija_y=ignicion_fija_y if not ajustar_ignicion else None
)

# Sincronizar después de completar la ejecución
cp.cuda.Stream.null.synchronize()
end_time = time.time()

print(f"Tiempo de ejecución en GPU: {end_time - start_time} segundos")