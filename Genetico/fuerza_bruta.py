import cupy as cp # type: ignore
import csv, os
from lectura_datos import preprocesar_datos
from operadores_geneticos import poblacion_inicial
from algoritmo import procesar_poblacion_batch

datos = preprocesar_datos()

vegetacion = datos["vegetacion"]
wx = datos["wx"]
wy = datos["wy"]
h_dx_mapa = datos["h_dx"]
h_dy_mapa = datos["h_dy"]
ny, nx = datos["ny"], datos["nx"]

############################## INCENDIO DE REFERENCIA ###############################################

incendio_referencia = cp.load("R_final.npy") # Cargar el archivo R_history.npy con NumPy

############################## PARÁMETROS DE LOS MAPAS ###############################################

d = 30 # Tamaño de cada celda
beta_veg = cp.where(vegetacion <= 2, 0, 0.1 * vegetacion) # Parámetros del modelo SI
gamma = cp.where(vegetacion <= 2, 0, 0.1) # Hacemos una máscara donde vegetación <=2, gamma >> 1/dt. Sino, vale 0.1. 
dt = 1/6 # Paso temporal.

############################## CONDICIÓN DE COURANT PARA LOS TÉRMINOS DIFUSIVOS Y ADVECTIVOS ###############################################

D_max = d**2 / (2*dt)
A_max = d / (cp.sqrt(2)*dt*cp.max(cp.sqrt(wx**2+wy**2)))
B_max = d / (cp.sqrt(2)*dt*cp.max(cp.sqrt(h_dx_mapa**2+h_dy_mapa**2)))

############################## PARÁMETROS DE LOS INCENDIOS SIMULADOS ###############################################

num_simulaciones = 20
cota = 0.95
limite_parametros = [(0, 200), (0, A_max * cota), (0, B_max * cota), (500, 900), (500, 900)]
combinaciones = poblacion_inicial(num_simulaciones, limite_parametros) 

############################## BARRIDA DE PARÁMETROS ###############################################

start = cp.cuda.Event()
end = cp.cuda.Event()

start.record()

num_steps = 1000
ruta_incendio_referencia = 'R_final.npy'
resultados = procesar_poblacion_batch(poblacion=combinaciones, ruta_incendio_referencia=ruta_incendio_referencia, 
                                      num_steps=num_steps, batch_size=20)

end.record()  # Marca el final en GPU
end.synchronize() # Sincroniza y mide el tiempo

gpu_time = cp.cuda.get_elapsed_time(start, end)  # Tiempo en milisegundos
print(f"Tiempo en GPU: {gpu_time:.3f} ms")


############################## GUARDADO DE RESULTADOS ###############################################

resultados_dir = f'Genetico/resultados_fuerza_bruta'
os.makedirs(resultados_dir, exist_ok=True)
# Obtener el task_id del SGE
task_id = os.environ.get('JOB_ID', 'default')

# Guardar resultados finales con información del task
final_csv_filename = os.path.join(resultados_dir, f'resultados_finales_task_{task_id}.csv')
with open(final_csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['D', 'A', 'B', 'x', 'y', 'fitness']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for resultado in resultados:
        writer.writerow(resultado)
    
print(f'Resultados guardados en: {resultados_dir}')
print(f'Task ID: {task_id}')


