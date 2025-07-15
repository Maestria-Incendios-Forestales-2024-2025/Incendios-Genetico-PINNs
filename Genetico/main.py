import cupy as cp # type: ignore
import csv, time
from config import d, dt, cota
from lectura_datos import preprocesar_datos
from algoritmo import genetic_algorithm

############################## CARGADO DE MAPAS #######################################################

datos = preprocesar_datos()
wx = datos["wx"]
wy = datos["wy"]
h_dx_mapa = datos["h_dx"]
h_dy_mapa = datos["h_dy"]

############################## CONDICIÓN DE COURANT PARA LOS TÉRMINOS DIFUSIVOS Y ADVECTIVOS ############

D_max = d**2 / (2*dt) # constante de difusión
A_max = d / (cp.sqrt(2)*dt*cp.max(cp.sqrt(wx**2+wy**2))) # constante de viento
B_max = d / (cp.sqrt(2)*dt*cp.max(cp.sqrt(h_dx_mapa**2+h_dy_mapa**2))) # contstante de pendiente

############################## EJECUCIÓN DEL ALGORITMO ###############################################

# Población aleatoria inicial (D, A, B, x, y)
limite_parametros = [(0, 200), (0, A_max * cota), (0, B_max * cota), (300, 720), (400, 800)]

# Sincronizar antes de empezar a medir el tiempo
cp.cuda.Stream.null.synchronize()
start_time = time.time()

# Ejecutar el GA con procesamiento en batch
resultados = genetic_algorithm(tamano_poblacion=10, generaciones=1, limite_parametros=limite_parametros, batch_size=5)

# Sincronizar después de completar la ejecución
cp.cuda.Stream.null.synchronize()
end_time = time.time()

print(f"Tiempo de ejecución en GPU: {end_time - start_time} segundos")

# Guardar los resultados en un archivo CSV
with open('resultados_genetico.csv', 'w', newline='') as csvfile:
    fieldnames = ['D', 'A', 'B', 'x', 'y', 'fitness']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for resultado in resultados:
        writer.writerow(resultado)

