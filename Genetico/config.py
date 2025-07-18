import cupy as cp # type: ignore

# Ruta de los archivos
ruta_mapas = ['c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin/ang_wind.asc',   # Dirección del viento
              'c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin/speed_wind.asc', # Velocidad del viento
              'c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin/asc_slope.asc',  # Pendiente del terreno
              'c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin/asc_CIEFAP.asc', # Vegetación
              'c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin/asc_aspect.asc', # Orientación del terreno
              'c:/Users/becer/OneDrive/Desktop/Maestría en Ciencias Físicas/Tesis/Incendios-Forestales---MCF-2024-2025/mapas_steffen_martin/area_quemada_SM.asc' # Área quemada de referencia
]

d = cp.float32(30) # Tamaño de cada celda
dt = cp.float32(1/6) # Intervalo de tiempo en horas
cota = 0.95 
num_steps = 10000 # Número de pasos de tiempo para la simulación
