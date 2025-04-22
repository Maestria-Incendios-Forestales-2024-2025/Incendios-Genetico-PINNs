import numpy as np
import cupy as cp # type: ignore
from modelo_rdc import spread_infection, courant
import csv
import argparse

############################## CARGADO DE MAPAS ###############################################

# Ruta de los archivos
ruta_mapas = ['/home/lucas.becerra/Incendios/mapas_steffen_martin/ang_wind.asc',   # Dirección del viento
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/speed_wind.asc', # Velocidad del viento
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/asc_slope.asc',  # Pendiente del terreno
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/asc_CIEFAP.asc', # Vegetación
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/asc_aspect.asc'] # Orientación del terreno

# Función para leer archivos .asc
def leer_asc(ruta):
    with open(ruta, 'r') as f:
        for i in range(6):  
            f.readline()
        data = np.loadtxt(f) 
        return cp.array(data, dtype=cp.float32) 

# Leer los mapas
datos = [leer_asc(mapa) for mapa in ruta_mapas]

# Asignar cada matriz a una variable
vientod = datos[0]
vientov = datos[1]
pendiente = datos[2]
vegetacion = datos[3]
orientacion = datos[4]

# Obtener dimensiones del mapa
ny, nx = vientod.shape 

############################## INCENDIO DE REFERENCIA ###############################################

# Cargar el archivo R_history.npy con NumPy
R_host = np.load("R_final.npy")

# Convertir a cupy
R = cp.asarray(R_host)

# Celdas quemadas en incendio de referencia y simulado
burnt_cells = cp.where(R > 0.5, 1, 0)

############################## PARÁMETROS DE LOS MAPAS ###############################################

d = 30 # Tamaño de cada celda
beta_veg = cp.where(vegetacion <= 2, 0, 0.1 * vegetacion) # Parámetros del modelo SI
gamma = cp.where(vegetacion <= 2, 100, 0.1) # Hacemos una máscara donde vegetación <=2, gamma >> 1/dt. Sino, vale 0.1. 
dt = 1/6 # Paso temporal.
# Transformación del viento a coordenadas cartesianas
wx = vientov * cp.cos(5/2 * cp.pi - vientod * cp.pi / 180) * 1000
wy = - vientov * cp.sin(5/2 * cp.pi - vientod * cp.pi / 180) * 1000
# Cálculo de la pendiente (usando mapas de pendiente y orientación)
h_dx_mapa = cp.tan(pendiente * cp.pi / 180) * cp.cos(orientacion * cp.pi / 180 - cp.pi/2)
h_dy_mapa = cp.tan(pendiente * cp.pi / 180) * cp.sin(orientacion * cp.pi / 180 - cp.pi/2)

############################## CONDICIÓN DE COURANT PARA LOS TÉRMINOS DIFUSIVOS Y ADVECTIVOS ###############################################

D_max = d**2 / (2*dt)
A_max = d / (cp.sqrt(2)*dt*cp.max(cp.sqrt(wx**2+wy**2)))
B_max = d / (cp.sqrt(2)*dt*cp.max(cp.sqrt(h_dx_mapa**2+h_dy_mapa**2)))

############################## PARÁMETROS DE LOS INCENDIOS SIMULADOS ###############################################

num_simulaciones = 1000
cota = 0.95

# Generar números aleatorios para los parámetros con un margen inferior al máximo usando una fracción del valor máximo
D_values = cp.random.uniform(0, 1000, num_simulaciones)  # Le puse D_max = 1000 
A_values = cp.random.uniform(0, A_max * cota, num_simulaciones)  # Reducir el rango superior en un 2%
B_values = cp.random.uniform(0, B_max * cota, num_simulaciones)  # Reducir el rango superior en un 2%

# Crear combinaciones de parámetros
combinaciones = cp.column_stack([
    D_values,
    A_values,
    B_values,
    cp.random.randint(500, 900, num_simulaciones),  # Coordenada x del punto de ignición
    cp.random.randint(500, 900, num_simulaciones),  # Coordenada y del punto de ignición
])

############################## BARRIDA DE PARÁMETROS ###############################################

num_steps = 1001

# Nombre del archivo de resultados y de checkpoint
parser = argparse.ArgumentParser(description="Ejecuta la simulación y guarda los resultados.")
parser.add_argument("archivo_salida", type=str, help="Nombre del archivo CSV para guardar los resultados.")
args = parser.parse_args()
archivo_principal = args.archivo_salida

#archivo_checkpoint = "checkpoint_2.csv"

# Abrir el archivo en modo escritura
with open(archivo_principal, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["D", "B", "A", "x", "y", "fitness"])  # Encabezado

    resultados_temp = []  # Lista temporal para los checkpoints

    # Recorrer y usar las combinaciones
    for idx, (D, A, B, x, y) in enumerate(combinaciones):

        # Chequeo que se cumpla la condición de courant
        while not courant(dt, D, A, B, d, wx, wy, h_dx=h_dx_mapa, h_dy=h_dy_mapa):
            # Seleccionar aleatoriamente qué parámetro reducir
            param_to_modify = cp.random.choice(["D", "A", "B"])

            if param_to_modify == "D":
                D *= cp.random.uniform(0.8, 0.99)
            elif param_to_modify == "A":
                A *= cp.random.uniform(0.8, 0.99)
            elif param_to_modify == "B":
                B *= cp.random.uniform(0.8, 0.99)

        while vegetacion[x.astype(cp.int32), y.astype(cp.int32)] <= 2:
            x, y = cp.random.randint(500, 900), cp.random.randint(500, 900)

        # Población inicial
        S_i = cp.ones((ny, nx))
        I_i = cp.zeros((ny, nx))
        R_i = cp.zeros((ny, nx))

        # Si hay combustible, encender fuego
        S_i[x.astype(cp.int32), y.astype(cp.int32)] = 0
        I_i[x.astype(cp.int32), y.astype(cp.int32)] = 1

        S_new_i = cp.empty_like(S_i)
        I_new_i = cp.empty_like(I_i)
        R_new_i = cp.empty_like(R_i)

        # Simulación
        for t in range(num_steps):
            spread_infection(S=S_i, I=I_i, R=R_i, S_new=S_new_i, I_new=I_new_i, R_new=R_new_i,
                             dt=dt, d=d, beta=beta_veg, gamma=gamma, 
                             D=D, wx=wx, wy=wy, h_dx=h_dx_mapa, h_dy=h_dy_mapa, A=A, B=B)
            
            S_i, S_new_i = S_new_i, S_i
            I_i, I_new_i = I_new_i, I_i
            R_i, R_new_i = R_new_i, R_i

            if not (cp.all((S_i >= 0) & (S_i <= 1)) and cp.all((I_i >= 0) & (I_i <= 1)) and cp.all((R_i >= 0) & (R_i <= 1))):
                break

        if not cp.all((R_i >= 0) & (R_i <= 1)): # La simulación explotó
            continue  # Pasa a la siguiente combinación sin guardar resultados

        # Celdas quemadas en incendio simulado
        burnt_cells_sim = cp.where(R_i > 0.5, 1, 0)

        # Si en la simulación no se quema ninguna celda, que no guarde ese resultado.
        if cp.sum(burnt_cells_sim) == 0:
            continue

        union = cp.sum((burnt_cells | burnt_cells_sim))
        interseccion = cp.sum((burnt_cells & burnt_cells_sim))
        
        fitness = (union - interseccion) / cp.sum(burnt_cells)

        # Guardar en archivo principal
        writer.writerow([D, B, A, x, y, fitness.get()])
        resultados_temp.append([D, B, A, x, y, fitness.get()])

        # Guardar checkpoint cada 1000 iteraciones
        # if idx % 1000 == 0 and idx > 0:
        #     with open(archivo_checkpoint, "w", newline="") as f_chk:
        #         writer_chk = csv.writer(f_chk)
        #         writer_chk.writerow(["D", "B", "A", "x", "y", "fitness"])  # Encabezado
        #         writer_chk.writerows(resultados_temp)  # Escribir los últimos 50 resultados
        #     resultados_temp = []  # Limpiar la lista temporal




