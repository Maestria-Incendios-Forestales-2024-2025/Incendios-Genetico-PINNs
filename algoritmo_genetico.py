import numpy as np
import cupy as cp # type: ignore
from math import sqrt
from modelo_rdc import spread_infection, courant
import csv
import random
import time

############################## CARGADO DE MAPAS ###############################################

# Ruta de los archivos
ruta_mapas = ['/home/lucas.becerra/Incendios/mapas_steffen_martin/ang_wind.asc',   # Dirección del viento
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/speed_wind.asc', # Velocidad del viento
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/asc_slope.asc',  # Pendiente del terreno
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/asc_CIEFAP.asc', # Vegetación
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/asc_aspect.asc', # Orientación del terreno
]

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
D = 50 # Coeficiente de difusión
beta_veg = cp.where(vegetacion <= 2, 0, 0.1 * vegetacion) # Parámetros del modelo SI
gamma = cp.where(vegetacion <= 2, 100, 0.1) # Hacemos una máscara donde vegetación <=2, gamma >> 1/dt. Sino, vale 0.1. 
dt = 1/6 # Paso temporal.
# Transformación del viento a coordenadas cartesianas
wx = vientov * cp.cos(5/2 * cp.pi - vientod * cp.pi / 180) * 1000
wy = - vientov * cp.sin(5/2 * cp.pi - vientod * cp.pi / 180) * 1000
# Constante A adimensional
A = 5e-4
# Constante B
B = 15
# Cálculo de la pendiente (usando mapas de pendiente y orientación)
h_dx_mapa = cp.tan(pendiente * cp.pi / 180) * cp.cos(orientacion * cp.pi / 180 - cp.pi/2)
h_dy_mapa = cp.tan(pendiente * cp.pi / 180) * cp.sin(orientacion * cp.pi / 180 - cp.pi/2)

############################## CONDICIÓN DE COURANT PARA LOS TÉRMINOS DIFUSIVOS Y ADVECTIVOS ###############################################

D_max = d**2 / (2*dt)
A_max = d / (cp.sqrt(2)*dt*cp.max(cp.sqrt(wx**2+wy**2)))
B_max = d / (cp.sqrt(2)*dt*cp.max(cp.sqrt(h_dx_mapa**2+h_dy_mapa**2)))

############################## ALGORITMO GENÉTICO ###############################################

def poblacion_inicial(tamano_poblacion, limite_parametros):
    poblacion = cp.array([cp.random.uniform(low, high, tamano_poblacion) for low, high in limite_parametros])
    return poblacion.T

def aptitud(D, A, B, x, y):

    num_steps = 1001

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

    # Población inicial de susceptibles e infectados
    S_i = cp.ones((ny, nx))  # Todos son susceptibles inicialmente
    I_i = cp.zeros((ny, nx)) # Ningún infectado al principio
    R_i = cp.zeros((ny, nx))

    # Si hay combustible, encender fuego
    S_i[x.astype(cp.int32), y.astype(cp.int32)] = 0
    I_i[x.astype(cp.int32), y.astype(cp.int32)] = 1

    # Iterar sobre las simulaciones
    for t in range(num_steps):
        S_i, I_i, R_i = spread_infection(S=S_i, I=I_i, R=R_i, dt=dt, d=d, beta=beta_veg, gamma=gamma, 
                                         D=D, wx=wx, wy=wy, h_dx=h_dx_mapa, h_dy=h_dy_mapa, A=A, B=B)
        if not (cp.all((R_i >= 0) & (R_i <= 1))):
            break

    if not cp.all((R_i >= 0) & (R_i <= 1)):
        return float('inf') # Pasa a la siguiente combinación sin guardar resultados

    # Celdas quemadas en el incendio simulado: si R_i > 0.5 esa celda está quemada
    burnt_cells_sim = cp.where(R_i > 0.5, 1, 0)

    union = cp.sum((burnt_cells | burnt_cells_sim))  # Celdas quemadas en al menos un mapa (unión)
    interseccion = cp.sum((burnt_cells & burnt_cells_sim))

    # Calcular el fitness
    fitness = (union - interseccion) / cp.sum(burnt_cells)

    return fitness

def tournament_selection(resultados, tournament_size=3):
    """Selecciona el individuo con mejor fitness dentro de un subconjunto aleatorio."""
    selected = random.sample(resultados, tournament_size)
    best_individual = min(selected, key=lambda x: x["fitness"])
    D, A, B, x, y = best_individual["D"], best_individual["A"], best_individual["B"], best_individual['x'], best_individual['y']
    return cp.array([D, A, B, x, y])

def crossover(parent1, parent2):
    """Realiza un cruce de un solo punto entre dos padres."""
    point = cp.random.randint(1, len(parent1))
    child1 = cp.concatenate((parent1[:point], parent2[point:]))
    child2 = cp.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def mutate(individual, mutation_rate, param_bounds):
    """Aplica una mutación aleatoria a los parámetros con una tasa dada."""
    for i in range(len(individual)):
        if cp.random.rand() < mutation_rate:
            individual[i] += cp.random.uniform(-0.1, 0.1) * (param_bounds[i][1] - param_bounds[i][0])
            individual[i] = cp.clip(individual[i], *param_bounds[i])
    return individual

def genetic_algorithm(tamano_poblacion, generations, param_bounds):
    """Implementa el algoritmo genético para estimar los parámetros del modelo de incendio."""
    combinaciones = poblacion_inicial(tamano_poblacion, limite_parametros)
    mutation_rate = 0.1

    resultados = []
    for D, A, B, x, y in combinaciones:
        fitness = aptitud(D, A, B, x, y)
        resultados.append({"D": D, "A": A, "B": B, "x": x, "y": y, "fitness": fitness})

    for gen in range(generations):
        new_population = []

        for _ in range(tamano_poblacion // 2): 
            parent1 = tournament_selection(resultados) # Selecciona 2 padres
            parent2 = tournament_selection(resultados)
            child1, child2 = crossover(parent1, parent2) # Se hace un crossover entre los padres y se generan 2 hijos
            child1 = mutate(child1, mutation_rate, param_bounds) # A esos hijos se les realiza una mutación
            child2 = mutate(child2, mutation_rate, param_bounds)
            new_population.extend([child1, child2]) # Estos hijos pasan a formar parte de la nueva población

        population = cp.array(new_population)
        resultados = []
        for D, A, B, x, y in population:
            fitness = aptitud(D, A, B, x, y)
            resultados.append({"D": D, "A": A, "B": B, "x": x, "y": y, "fitness": fitness})

        best_fitness = min(resultados, key=lambda x: x["fitness"])["fitness"]
        print(f'Generación {gen+1}: Mejor fitness = {best_fitness}')

        # Opcional: reducir la tasa de mutación con el tiempo
        mutation_rate *= 0.99

    return resultados

############################## ALGORITMO GENÉTICO ###############################################

cota = 0.95

# Población aleatoria inicial
limite_parametros = [(0, 1000), (0, A_max * cota), (0, B_max * cota), (500, 900), (500, 900)]

# Sincronizar antes de empezar a medir el tiempo
cp.cuda.Stream.null.synchronize()
start_time = time.time()

# Ejecutar el GA
resultados = genetic_algorithm(tamano_poblacion=100, generations=100, param_bounds=limite_parametros)

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

