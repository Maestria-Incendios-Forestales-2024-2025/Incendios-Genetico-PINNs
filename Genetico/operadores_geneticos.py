import cupy as cp # type: ignore
import random

############################## ARMADO DE POBLACIÓN INICIAL ###############################################

def poblacion_inicial(tamano_poblacion, limite_parametros):
    """
    Genera población inicial con parámetros básicos + parámetros de vegetación.
    
    Args:
        tamano_poblacion: Número de individuos
        limite_parametros: Lista de tuplas (min, max) para cada parámetro
    
    Returns:
        Array de CuPy con individuos (cada fila es un individuo)
    """
    # Generar población aleatoria para todos los parámetros
    poblacion = []
    
    for _ in range(tamano_poblacion):
        individuo = []
        for i, (low, high) in enumerate(limite_parametros):
            valor = cp.random.uniform(low, high)
            # Generar valor aleatorio entre low y high
            if i > 9 and i <= 14:
                while valor > individuo[i - 5]: # evitar que beta sea menor que gamma
                    valor = cp.random.uniform(low, high)
            individuo.append(float(valor))
        poblacion.append(individuo)
    
    return cp.array(poblacion, dtype=cp.float32)

############################## SELECCIÓN DE TORNEO ###############################################

def tournament_selection(resultados, tournament_size=3):
    """Selecciona el individuo con mejor fitness dentro de un subconjunto aleatorio."""
    selected = random.sample(resultados, tournament_size)
    best_individual = min(selected, key=lambda x: x["fitness"])
    
    # Extraer parámetros básicos
    D, A, B, x, y = best_individual["D"], best_individual["A"], best_individual["B"], best_individual['x'], best_individual['y']
    
    # Extraer parámetros de vegetación (si existen)
    betas = best_individual.get('betas', [])
    gammas = best_individual.get('gammas', [])
    
    # Crear array plano concatenando todos los parámetros
    basic_params = [D, A, B, x, y]
    
    # Convertir betas y gammas a listas planas si no lo son
    if isinstance(betas, (list, tuple)):
        beta_list = list(betas)
    else:
        beta_list = [betas]
    
    if isinstance(gammas, (list, tuple)):
        gamma_list = list(gammas)
    else:
        gamma_list = [gammas]
    
    # Concatenar todos los parámetros en un solo array plano
    all_params = basic_params + beta_list + gamma_list
    
    return cp.array(all_params, dtype=cp.float32)

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