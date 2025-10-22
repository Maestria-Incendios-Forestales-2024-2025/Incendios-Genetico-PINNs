import cupy as cp # type: ignore
import random

############################## ARMADO DE POBLACIÓN INICIAL ###############################################

def poblacion_inicial(tamano_poblacion, limite_parametros):
    """
    Genera población inicial dentro de los límites establecidos
    Args:
        tamano_poblacion: Número de individuos
        limite_parametros: Lista de tuplas (min, max) para cada parámetro
    Returns:
        Array de CuPy con individuos (cada fila es un individuo)
    """
    n_params = len(limite_parametros)
    
    # Genera números aleatorios uniformes en [0,1]
    rs = cp.random.default_rng(seed=42)

    rand = rs.random((tamano_poblacion, n_params), dtype=cp.float32)

    # Convierte límites a arrays
    lows  = cp.array([low for low, _ in limite_parametros], dtype=cp.float32)
    highs = cp.array([high for _, high in limite_parametros], dtype=cp.float32)
    
    # Escala cada columna al rango correspondiente
    poblacion = lows + rand * (highs - lows)
    
    return poblacion

############################## SELECCIÓN DE TORNEO ###############################################

def tournament_selection(resultados, tournament_size=3, ajustar_beta_gamma=True, ajustar_ignicion=True):
    """Selecciona el individuo con mejor fitness dentro de un subconjunto aleatorio."""
    selected = random.sample(resultados, tournament_size)
    best_individual = min(selected, key=lambda x: x["fitness"])
    
    # Extraer parámetros básicos
    D, A, B = best_individual["D"], best_individual["A"], best_individual["B"]
    all_params = cp.array([D, A, B], dtype=cp.float32)

    # Extraer x, y si corresponde
    if ajustar_ignicion or not ajustar_beta_gamma:
        x, y = best_individual['x'], best_individual['y']
        all_params = cp.concatenate([all_params, cp.array([x], dtype=cp.float32), cp.array([y], dtype=cp.float32)])

    # Extraer betas y gammas si corresponde
    if ajustar_beta_gamma:
        if best_individual['betas'].size > 1:
            betas = best_individual.get('betas', cp.array([], dtype=cp.float32))
            gammas = best_individual.get('gammas', cp.array([], dtype=cp.float32))
            all_params = cp.concatenate([all_params, cp.asarray(betas, dtype=cp.float32), cp.asarray(gammas, dtype=cp.float32)])
        else:
            betas = best_individual['betas']
            gammas = best_individual['gammas']
            all_params = cp.append(all_params, [betas, gammas])
    
    return all_params

############################## CROSSOVER #########################################################

def crossover(parent1, parent2):
    """Realiza un cruce de un solo punto entre dos padres."""
    # Asegurar que ambos padres tengan la misma longitud
    min_length = min(len(parent1), len(parent2))
    
    # Punto de cruce aleatorio (evitar los extremos)
    point = int(cp.random.randint(1, min_length))
    
    # Crear hijos intercambiando segmentos
    child1 = cp.concatenate((parent1[:point], parent2[point:min_length]))
    child2 = cp.concatenate((parent2[:point], parent1[point:min_length]))
    
    # Si los padres tenían longitudes diferentes, completar con el padre más largo
    if len(parent1) > min_length:
        child1 = cp.concatenate((child1, parent1[min_length:]))
    if len(parent2) > min_length:
        child2 = cp.concatenate((child2, parent2[min_length:]))
    
    return child1, child2

############################## MUTACIÓN #########################################################

def mutate(individual, mutation_rate, param_bounds):
    """Aplica una mutación aleatoria a los parámetros con una tasa dada."""
    # Crear una copia del individuo para no modificar el original
    mutated = cp.copy(individual)
    
    # Aplicar mutación a cada parámetro
    for i in range(len(mutated)):
        if cp.random.rand() < mutation_rate:
            # Verificar que tenemos límites para este parámetro
            if i < len(param_bounds):
                low, high = param_bounds[i]
                # Mutación gaussiana con límites
                mutation_strength = 0.1 * (high - low)
                mutation = cp.random.normal(0, mutation_strength)
                mutated[i] = mutated[i] + mutation
                # Aplicar límites
                mutated[i] = cp.clip(mutated[i], low, high)
            else:
                # Si no hay límites definidos, mutación pequeña
                mutated[i] += cp.random.uniform(-0.01, 0.01)
                
    return mutated