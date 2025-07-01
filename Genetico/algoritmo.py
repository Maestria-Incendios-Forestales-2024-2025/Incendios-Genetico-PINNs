import cupy as cp  # type: ignore
from operadores_geneticos import poblacion_inicial, tournament_selection, crossover, mutate
from fitness import aptitud

############################## ALGORITMO GENÉTICO #########################################################

def genetic_algorithm(tamano_poblacion, generaciones, limite_parametros):
    """Implementa el algoritmo genético para estimar los parámetros del modelo de incendio."""
    combinaciones = poblacion_inicial(tamano_poblacion, limite_parametros)
    mutation_rate = 0.3

    resultados = []
    for i, individuo in enumerate(combinaciones):
        D, A, B, x, y = individuo
        D, A, B, x, y = D.item(), A.item(), B.item(), int(x.item()), int(y.item())  # Convertir a tipos nativos de Python
        fitness = aptitud(D, A, B, x, y)
        print(f'Individuo {i+1}: D={D}, A={A}, B={B}, x={x}, y={y}, fitness={fitness}')
        resultados.append({"D": D, "A": A, "B": B, "x": x, "y": y, "fitness": fitness})

    #print(resultados)
    print(f'Generación 0: Mejor fitness = {min(resultados, key=lambda x: x["fitness"])["fitness"]}')

    for gen in range(generaciones):
        print(f'Iniciando generación {gen+1}...')
        new_population = []

        # Elitismo
        elite = min(resultados, key=lambda x: x["fitness"])

        for _ in range(tamano_poblacion // 2): 
            parent1 = tournament_selection(resultados) # Selecciona 2 padres
            parent2 = tournament_selection(resultados)
            child1, child2 = crossover(parent1, parent2) # Se hace un crossover entre los padres y se generan 2 hijos
            child1 = mutate(child1, mutation_rate, limite_parametros) # A esos hijos se les realiza una mutación
            child2 = mutate(child2, mutation_rate, limite_parametros)
            new_population.extend([child1, child2]) # Estos hijos pasan a formar parte de la nueva población

        population = cp.array(new_population)
        resultados = []
        for i, individuo in enumerate(population):
            D, A, B, x, y = individuo
            D, A, B, x, y = D.item(), A.item(), B.item(), int(x.item()), int(y.item())  # Convertir a tipos nativos de Python
            fitness = aptitud(D, A, B, x, y)
            print(f'Individuo {i+1}: D={D}, A={A}, B={B}, x={x}, y={y}, fitness={fitness}')
            resultados.append({"D": D, "A": A, "B": B, "x": x, "y": y, "fitness": fitness})

        peor_idx = max(range(len(resultados)), key=lambda i: resultados[i]["fitness"])
        resultados[peor_idx] = elite  # Mantener el mejor individuo de la generación anterior

        best_fitness = min(resultados, key=lambda x: x["fitness"])["fitness"]
        print(f'Generación {gen+1}: Mejor fitness = {best_fitness}')

        # Opcional: reducir la tasa de mutación con el tiempo
        mutation_rate *= 0.99

    return resultados