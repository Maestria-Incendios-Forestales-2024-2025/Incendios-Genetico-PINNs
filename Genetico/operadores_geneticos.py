import cupy as cp # type: ignore
import random

############################## ARMADO DE POBLACIÓN INICIAL ###############################################

def poblacion_inicial(tamano_poblacion, limite_parametros):
    poblacion = cp.array([cp.random.uniform(low, high, tamano_poblacion) for low, high in limite_parametros], dtype=cp.float32)
    return poblacion.T

############################## SELECCIÓN DE TORNEO ###############################################

def tournament_selection(resultados, tournament_size=3):
    """Selecciona el individuo con mejor fitness dentro de un subconjunto aleatorio."""
    selected = random.sample(resultados, tournament_size)
    best_individual = min(selected, key=lambda x: x["fitness"])
    D, A, B, x, y = best_individual["D"], best_individual["A"], best_individual["B"], best_individual['x'], best_individual['y']
    return cp.array([D, A, B, x, y])

############################## CROSSOVER #########################################################

def crossover(parent1, parent2):
    """Realiza un cruce de un solo punto entre dos padres."""
    point = cp.random.randint(1, len(parent1))
    child1 = cp.concatenate((parent1[:point], parent2[point:]))
    child2 = cp.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

############################## MUTACIÓN #########################################################

def mutate(individual, mutation_rate, param_bounds):
    """Aplica una mutación aleatoria a los parámetros con una tasa dada."""
    for i in range(len(individual)):
        if cp.random.rand() < mutation_rate:
            individual[i] += cp.random.uniform(-0.1, 0.1) * (param_bounds[i][1] - param_bounds[i][0])
            individual[i] = cp.clip(individual[i], *param_bounds[i])
    return individual