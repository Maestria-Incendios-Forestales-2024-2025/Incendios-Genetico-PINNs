# Ruta de los archivos
ruta_mapas = ['/home/lucas.becerra/Incendios/mapas_steffen_martin/ang_wind.asc',   # Dirección del viento
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/speed_wind.asc', # Velocidad del viento
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/asc_slope.asc',  # Pendiente del terreno
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/asc_CIEFAP.asc', # Vegetación
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/asc_aspect.asc', # Orientación del terreno
]

d = 30 # Tamaño de cada celda
D = 50 # Coeficiente de difusión
A = 5e-4 # Coeficiente de advección por viento
B = 15 # Coeficiente de advección por pendiente 
cota = 0.95 
num_steps = 1001 # Número de pasos de tiempo para la simulación
