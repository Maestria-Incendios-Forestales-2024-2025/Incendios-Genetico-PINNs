import cupy as cp  # type: ignore
import socket
import os

# Detectar el entorno según el hostname
hostname = socket.gethostname()

if "rocks7frontend" in hostname or "compute" in hostname:
    # Cluster
    base_path = "/home/lucas.becerra/Incendios-Genetico-PINNs/mapas_steffen_martin"
elif "ccad.unc.edu.ar" in hostname:
    # CCAD
    base_path = "/home/lbecerra/Incendios-Genetico-PINNs/mapas_steffen_martin"
else:
    # Local
    base_path = "c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin"

# Construcción de rutas
ruta_mapas = [
    os.path.join(base_path, "ang_wind.asc"),    # Dirección del viento
    os.path.join(base_path, "speed_wind.asc"),  # Velocidad del viento
    os.path.join(base_path, "asc_slope.asc"),   # Pendiente del terreno
    os.path.join(base_path, "asc_CIEFAP.asc"),  # Vegetación
    os.path.join(base_path, "asc_aspect.asc"),  # Orientación del terreno
]

# Parámetros comunes
d = cp.float32(30)   # Tamaño de cada celda
dt = cp.float32(1/2) # Intervalo de tiempo en horas
cota = 0.95
