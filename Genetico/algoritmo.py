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
            # print(f'Seleccionando al primer padre para la generación {gen+1}...')
            parent1 = tournament_selection(resultados) # Selecciona 2 padres
            # print(f'Padre seleccionado: D={parent1[0]}, A={parent1[1]}, B={parent1[2]}, x={parent1[3]}, y={parent1[4]}')
            # print(f'Seleccionando al segundo padre para la generación {gen+1}...')
            parent2 = tournament_selection(resultados)
            # print(f'Padre seleccionado: D={parent2[0]}, A={parent2[1]}, B={parent2[2]}, x={parent2[3]}, y={parent2[4]}')
            child1, child2 = crossover(parent1, parent2) # Se hace un crossover entre los padres y se generan 2 hijos
            # print(f'Hijos generados: D={child1[0]}, A={child1[1]}, B={child1[2]}, x={child1[3]}, y={child1[4]} \
                #    y D={child2[0]}, A={child2[1]}, B={child2[2]}, x={child2[3]}, y={child2[4]}')
            child1 = mutate(child1, mutation_rate, limite_parametros) # A esos hijos se les realiza una mutación
            child2 = mutate(child2, mutation_rate, limite_parametros)
            # print(f'Hijos mutados: D={child1[0]}, A={child1[1]}, B={child1[2]}, x={child1[3]}, y={child1[4]} \
                #    y D={child2[0]}, A={child2[1]}, B={child2[2]}, x={child2[3]}, y={child2[4]}')
            new_population.extend([child1, child2]) # Estos hijos pasan a formar parte de la nueva población

        population = cp.array(new_population)
        resultados = []
        for i, individuo in enumerate(population):
            D, A, B, x, y = individuo
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