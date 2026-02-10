import cupy as cp  # type: ignore
import os
from operadores_geneticos import poblacion_inicial, tournament_selection, crossover, mutate
from fitness import FitnessEvaluator
from lectura_datos import cargar_poblacion_preentrenada, leer_incendio_referencia, guardar_resultados
import socket

hostname = socket.gethostname()

############################## PROCESAMIENTO EN BATCH ###############################################

def procesar_poblacion_batch(poblacion, ruta_incendio_referencia, limite_parametros, ctx, num_steps=10000, batch_size=10, 
                             ajustar_beta_gamma=True, beta_fijo=None, gamma_fijo=None, ajustar_ignicion=True,
                             ignicion_fija_x=None, ignicion_fija_y=None):
    """
    Procesa una población en batches para aprovechar el paralelismo.
    
    Args:
        poblacion: Lista de individuos (D, A, B, x, y)
        batch_size: Tamaño del batch para procesamiento en paralelo
    
    Returns:
        Lista de resultados con fitness calculado
    """  
    resultados = []
    evaluator = FitnessEvaluator(ctx)

    incendio_referencia = leer_incendio_referencia(ruta_incendio_referencia)
    celdas_quemadas_referencia = cp.where(incendio_referencia > 0.001, 1, 0)

    for i in range(0, len(poblacion), batch_size):
        batch = poblacion[i:i+batch_size]

        print(f'Procesando batch {i//batch_size + 1} de {len(poblacion) // batch_size}...')

        parametros_batch = []
        for individuo in batch:
            D, A, B = individuo[:3]
            D, A, B = D.item(), A.item(), B.item()

            if ajustar_beta_gamma and ajustar_ignicion:         # Exp2
                x, y = int(individuo[3].item()), int(individuo[4].item())
                betas = individuo[5]  # beta_veg
                gammas = individuo[6]  # gamma
                parametros_batch.append((D, A, B, x, y, betas, gammas))
            elif ajustar_beta_gamma and not ajustar_ignicion:   # Exp3
                betas = individuo[3:8]
                gammas = individuo[8:13]
                parametros_batch.append((D, A, B, betas, gammas))
            else:                                               # Exp1
                x, y = int(individuo[3].item()), int(individuo[4].item())
                parametros_batch.append((D, A, B, x, y))

        parametros_validados = []

        if ajustar_beta_gamma and ajustar_ignicion:             # Exp2
            for D, A, B, x, y, betas, gammas in parametros_batch:
                A, B = evaluator.validate_courant_and_adjust(A, B)
                x, y = evaluator.validate_ignition_point(x, y, incendio_referencia, limite_parametros)
                betas, gammas = evaluator.validate_beta_gamma(betas, gammas)
                parametros_validados.append((D, A, B, x, y, betas, gammas))
        elif ajustar_beta_gamma and not ajustar_ignicion:       # Exp3 
            for D, A, B, betas, gammas in parametros_batch:
                A, B = evaluator.validate_courant_and_adjust(A, B)
                betas, gammas = evaluator.validate_beta_gamma(betas, gammas)
                parametros_validados.append((D, A, B, betas, gammas))
        else:                                                   # Exp1
            for D, A, B, x, y in parametros_batch:
                A, B = evaluator.validate_courant_and_adjust(A, B)
                x, y = evaluator.validate_ignition_point(x, y, incendio_referencia, limite_parametros)
                parametros_validados.append((D, A, B, x, y))

        fitness_values = evaluator.evaluate_batch(parametros_validados, celdas_quemadas_referencia, num_steps, 
                                       ajustar_beta_gamma=ajustar_beta_gamma, beta_fijo=beta_fijo, gamma_fijo=gamma_fijo, 
                                       ajustar_ignicion=ajustar_ignicion, ignicion_fija_x=ignicion_fija_x, 
                                       ignicion_fija_y=ignicion_fija_y)

        if ajustar_beta_gamma and ajustar_ignicion:              # Exp2
            for params, fitness in zip(parametros_validados, fitness_values):
                D, A, B, x, y, betas, gammas = params
                resultados.append({
                    "D": D, "A": A, "B": B, "x": x, "y": y, "fitness": fitness,
                    "betas": betas, "gammas": gammas
                })
        elif ajustar_beta_gamma and not ajustar_ignicion:        # Exp3
            for params, fitness in zip(parametros_validados, fitness_values):
                D, A, B, betas, gammas = params
                resultados.append({
                    "D": D, "A": A, "B": B, "fitness": fitness,
                    "betas": betas, "gammas": gammas
                })
        else:                                                     # Exp1
            for params, fitness in zip(parametros_validados, fitness_values):
                D, A, B, x, y = params
                resultados.append({
                    "D": D, "A": A, "B": B, "x": x, "y": y, "fitness": fitness 
                })

    return resultados

############################## ALGORITMO GENÉTICO #########################################################

def genetic_algorithm(tamano_poblacion, generaciones, limite_parametros, ruta_incendio_referencia, ctx,
                      archivo_preentrenado=None, generacion_preentrenada=0, num_steps=10000, batch_size=10,
                      ajustar_beta_gamma=True, beta_fijo=None, gamma_fijo=None, ajustar_ignicion=True, 
                      ignicion_fija_x=None, ignicion_fija_y=None):
    
    """Implementa el algoritmo genético para estimar los parámetros del modelo de incendio."""
        
    resultados_dir = f'resultados'
    os.makedirs(resultados_dir, exist_ok=True)

    # Si hay una población preentrenada la carga, sino se genera una nueva población inicial
    if archivo_preentrenado:
        resultados = cargar_poblacion_preentrenada(archivo_preentrenado, tamano_poblacion, limite_parametros,
                                                   ajustar_beta_gamma=ajustar_beta_gamma, ajustar_ignicion=ajustar_ignicion)
    else:
        combinaciones = poblacion_inicial(tamano_poblacion, limite_parametros)
        resultados = procesar_poblacion_batch(combinaciones, ruta_incendio_referencia, limite_parametros, ctx,
                                              num_steps=num_steps, batch_size=batch_size, 
                                              ajustar_beta_gamma=ajustar_beta_gamma, 
                                              beta_fijo=beta_fijo, gamma_fijo=gamma_fijo, ajustar_ignicion=ajustar_ignicion,
                                              ignicion_fija_x=ignicion_fija_x, ignicion_fija_y=ignicion_fija_y)
 
    mutation_rate = 0.3 * 0.99**generacion_preentrenada

    for i, individuo in enumerate(resultados, 1):
        if ajustar_beta_gamma and ajustar_ignicion:   # Exp2
            print(
                f'Individuo {i}: D={individuo["D"]}, A={individuo["A"]}, B={individuo["B"]}, x={individuo["x"]}, y={individuo["y"]}, \n'
                f'\t beta={individuo["betas"]}, \n'
                f'\t gamma={individuo["gammas"]}, \n'
                f'\t fitness={individuo["fitness"]:.4f}'
            )
        elif ajustar_beta_gamma and not ajustar_ignicion:  # Exp3
            print(
                f'Individuo {i}: D={individuo["D"]}, A={individuo["A"]}, B={individuo["B"]}, \n'
                f'\t beta={individuo["betas"]}, \n'
                f'\t gamma={individuo["gammas"]}, \n'
                f'\t fitness={individuo["fitness"]:.4f}'
            )
        else:                                             # Exp1
            print(
                f'Individuo {i}: D={individuo["D"]}, A={individuo["A"]}, B={individuo["B"]}, x={individuo["x"]}, y={individuo["y"]}, \n'
                f'\t fitness={individuo["fitness"]:.4f}'
            )

    guardar_resultados(resultados, resultados_dir, -1+generacion_preentrenada, 
                       ajustar_beta_gamma=ajustar_beta_gamma, ajustar_ignicion=ajustar_ignicion)
    print(f'Generación 0: Mejor fitness = {min(resultados, key=lambda x: x["fitness"])["fitness"]}')

    for gen in range(generaciones):
        print(f'Iniciando generación {gen+1}...')
        new_population = []
        elite = min(resultados, key=lambda x: x["fitness"])

        for _ in range(tamano_poblacion // 2):
            parent1 = tournament_selection(resultados, ajustar_beta_gamma=ajustar_beta_gamma, ajustar_ignicion=ajustar_ignicion)
            parent2 = tournament_selection(resultados, ajustar_beta_gamma=ajustar_beta_gamma, ajustar_ignicion=ajustar_ignicion)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, limite_parametros)
            child2 = mutate(child2, mutation_rate, limite_parametros)
            new_population.extend([child1, child2])

        population = cp.array(new_population)

        print(f"Procesando generación {gen+1} en batch...")
        resultados = procesar_poblacion_batch(population, ruta_incendio_referencia, limite_parametros, ctx,
                                              num_steps=num_steps, batch_size=batch_size,
                                              ajustar_beta_gamma=ajustar_beta_gamma, 
                                              beta_fijo=beta_fijo, gamma_fijo=gamma_fijo, ajustar_ignicion=ajustar_ignicion,
                                              ignicion_fija_x=ignicion_fija_x, ignicion_fija_y=ignicion_fija_y)

        peor_idx = max(range(len(resultados)), key=lambda i: resultados[i]["fitness"])
        resultados[peor_idx] = elite

        best_fitness = min(resultados, key=lambda x: x["fitness"])["fitness"]
        print(f'Generación {gen+1}: Mejor fitness = {best_fitness}')

        mutation_rate *= 0.99

        for i, individuo in enumerate(resultados, 1):
            if ajustar_beta_gamma and ajustar_ignicion:
                print(
                    f'Individuo {i}: D={individuo["D"]}, A={individuo["A"]}, B={individuo["B"]}, x={individuo["x"]}, y={individuo["y"]}, \n'
                    f'\t beta={individuo["betas"]}, \n'
                    f'\t gamma={individuo["gammas"]}, \n'
                    f'\t fitness={individuo["fitness"]:.4f}'
                )
            elif ajustar_beta_gamma and not ajustar_ignicion:
                print(
                    f'Individuo {i}: D={individuo["D"]}, A={individuo["A"]}, B={individuo["B"]}, \n'
                    f'\t beta={individuo["betas"]}, \n'
                    f'\t gamma={individuo["gammas"]}, \n'
                    f'\t fitness={individuo["fitness"]:.4f}'
                )
            else:
                print(
                    f'Individuo {i}: D={individuo["D"]}, A={individuo["A"]}, B={individuo["B"]}, x={individuo["x"]}, y={individuo["y"]}, \n'
                    f'\t fitness={individuo["fitness"]:.4f}'
                )

        guardar_resultados(resultados, resultados_dir, gen+generacion_preentrenada, 
                           ajustar_beta_gamma=ajustar_beta_gamma, ajustar_ignicion=ajustar_ignicion)

    print(f'Resultados guardados en: {resultados_dir}')

    return resultados