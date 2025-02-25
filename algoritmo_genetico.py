import numpy as np
import cupy as cp # type: ignore
from math import sqrt
from modelo_rdc import spread_infection
import csv
import random

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


############################## ALGORITMO GENÉTICO ###############################################

def poblacion_inicial(tamaño_poblacion, limite_parametros):
    poblacion = np.array([np.random.uniform(low, high, tamaño_poblacion) for low, high in limite_parametros])
    return poblacion.T

def aptitud(D, A, B):

    num_steps = 1001

    # Población inicial de susceptibles e infectados
    S_i = cp.ones((ny, nx))  # Todos son susceptibles inicialmente
    I_i = cp.zeros((ny, nx)) # Ningún infectado al principio
    R_i = cp.zeros((ny, nx))

    # Infectados en el centro de la grilla
    S_i[700, 700] = 0
    I_i[700, 700] = 1

    # Iterar sobre las simulaciones
    for t in range(num_steps):
        S_i, I_i, R_i = spread_infection(S=S_i, I=I_i, R=R_i, dt=dt, d=d, beta=beta_veg, gamma=gamma, D=D, wx=wx, wy=wy, h_dx=h_dx_mapa, h_dy=h_dy_mapa, A=A, B=B)

    # Celdas quemadas en el incendio de referencia: si R > 0.5 esa celda está quemada
    burnt_cells = np.where(R > 0.5, 1, 0)

    # Celdas quemadas en el incendio simulado: si R_i > 0.5 esa celda está quemada
    burnt_cells_sim = np.where(R_i > 0.5, 1, 0)

    union = cp.sum((burnt_cells | burnt_cells_sim))  # Celdas quemadas en al menos un mapa (unión)
    interseccion = cp.sum((burnt_cells & burnt_cells_sim))

    # Calcular el fitness
    fitness = (union - interseccion) / cp.sum(burnt_cells)
    #print(fitness)

    return fitness

def tournament_selection(resultados, tournament_size=3):
    """Selecciona el individuo con mejor fitness dentro de un subconjunto aleatorio."""
    selected = random.sample(resultados, tournament_size)
    best_individual = min(selected, key=lambda x: x["fitness"])
    return best_individual["D"], best_individual["A"], best_individual["B"]

def crossover(parent1, parent2):
    """Realiza un cruce de un solo punto entre dos padres."""
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def mutate(individual, mutation_rate, param_bounds):
    """Aplica una mutación aleatoria a los parámetros con una tasa dada."""
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.uniform(-0.1, 0.1) * (param_bounds[i][1] - param_bounds[i][0])
            individual[i] = np.clip(individual[i], *param_bounds[i])
    return individual

def genetic_algorithm(pop_size, generations, param_bounds):
    """Implementa el algoritmo genético para estimar los parámetros del modelo de incendio."""
    combinaciones = poblacion_inicial(tamaño_poblacion, limite_parametros)
    mutation_rate = 0.1

    resultados = []
    for D, A, B in combinaciones:
        fitness = aptitud(D, A, B)
        resultados.append({"D": D, "A": A, "B": B, "fitness": fitness})

    for gen in range(generations):
        new_population = []

        for _ in range(pop_size // 2):
            parent1 = tournament_selection(resultados)
            parent2 = tournament_selection(resultados)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, param_bounds)
            child2 = mutate(child2, mutation_rate, param_bounds)
            new_population.extend([child1, child2])

        population = np.array(new_population)
        resultados = []
        for D, A, B in population:
            fitness = aptitud(D, A, B)
            resultados.append({"D": D, "A": A, "B": B, "fitness": fitness})

        best_fitness = min(resultados, key=lambda x: x["fitness"])["fitness"]
        print(f'Generación {gen+1}: Mejor fitness = {best_fitness}')

        # Opcional: reducir la tasa de mutación con el tiempo
        mutation_rate *= 0.99

    best_individual = min(resultados, key=lambda x: x["fitness"])
    return best_individual["D"], best_individual["A"], best_individual["B"], best_individual["fitness"]

# Población aleatoria inicial
limite_parametros = [(30, 50), (0.0003, 0.0007), (5, 25)]
tamaño_poblacion = 100

# Ejecutar el GA
best_D, best_A, best_B, best_fit = genetic_algorithm(pop_size=50, generations=100, param_bounds=limite_parametros)
print("Mejores parámetros encontrados:", best_D, best_A, best_B, "Fitness:", best_fit)
