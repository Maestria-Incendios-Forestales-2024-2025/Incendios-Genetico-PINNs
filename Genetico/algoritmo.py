import cupy as cp  # type: ignore
import csv, os, sys
from operadores_geneticos import poblacion_inicial, tournament_selection, crossover, mutate
from fitness import aptitud_batch
from fitness import aptitud_batch
from config import d, dt
from lectura_datos import preprocesar_datos, cargar_poblacion_preentrenada, leer_incendio_referencia

# Agregar el directorio padre al path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelo_rdc import courant

############################## CARGADO DE MAPAS ###############################################

# Cargar datos necesarios para validación
datos = preprocesar_datos()
vegetacion = datos["vegetacion"]
wx = datos["wx"]
wy = datos["wy"]
h_dx_mapa = datos["h_dx"]
h_dy_mapa = datos["h_dy"]

############################## CHEQUEO DE CONDICIÓN DE COURANT ###############################################

def validate_courant_and_adjust(D, A, B):
    """Valida la condición de Courant y ajusta parámetros si es necesario."""
    
    iteraciones = 0
    while not courant(dt, D, A, B, d, wx, wy, h_dx=h_dx_mapa, h_dy=h_dy_mapa):
        iteraciones += 1
        # Alternativa más eficiente: seleccionar aleatoriamente entre 0, 1, 2
        param_idx = int(cp.random.randint(0, 3))  # 0, 1, o 2
        
        if param_idx == 0:  # D
            D = float(D * float(cp.random.uniform(0.8, 0.99)))
        elif param_idx == 1:  # A
            A = float(A * float(cp.random.uniform(0.8, 0.99)))
        elif param_idx == 2:  # B
            B = float(B * float(cp.random.uniform(0.8, 0.99)))
        
        # Evitar bucles infinitos
        if iteraciones > 100:
            print(f"Warning: Validación Courant tomó {iteraciones} iteraciones")
            break
    return D, A, B

############################## VALIDACIÓN DE PUNTO DE IGNICIÓN ###############################################

def validate_ignition_point(x, y, incendio_referencia, limite_parametros):
    """Valida que el punto de ignición tenga combustible."""
    lim_x, lim_y = limite_parametros[3], limite_parametros[4]
    while vegetacion[int(y), int(x)] <= 2 or incendio_referencia[int(y), int(x)] <= 0.001:
        x, y = float(cp.random.randint(lim_x[0], lim_x[1])), float(cp.random.randint(lim_y[0], lim_y[1]))
    return x, y

############################## PROCESAMIENTO EN BATCH ###############################################

def procesar_poblacion_batch(poblacion, ruta_incendio_referencia, limite_parametros, num_steps=10000, batch_size=10):
    """
    Procesa una población en batches para aprovechar el paralelismo.
    
    Args:
        poblacion: Lista de individuos (D, A, B, x, y)
        batch_size: Tamaño del batch para procesamiento en paralelo
    
    Returns:
        Lista de resultados con fitness calculado
    """  
    resultados = []

    incendio_referencia = leer_incendio_referencia(ruta_incendio_referencia)
    celdas_quemadas_referencia = cp.where(incendio_referencia > 0.001, 1, 0)
    
    for i in range(0, len(poblacion), batch_size):
        batch = poblacion[i:i+batch_size]
        
        print(f'Procesando batch {i//batch_size + 1} de {len(poblacion) // batch_size}...')
        
        # Preparar parámetros para el batch
        parametros_batch = []
        for individuo in batch:
            D, A, B, x, y = individuo[:5]
            betas = individuo[5:10]  # beta_veg
            gammas = individuo[10:15]  # gamma
            
            D, A, B, x, y = D.item(), A.item(), B.item(), int(x.item()), int(y.item())
            betas = betas.tolist()
            gammas = gammas.tolist()

            parametros_batch.append((D, A, B, x, y, betas, gammas))
        
        # Validar parámetros
        parametros_validados = []
        for D, A, B, x, y, betas, gammas in parametros_batch:
            # Validar y ajustar parámetros
            D, A, B = validate_courant_and_adjust(D, A, B)
            x, y = validate_ignition_point(x, y, incendio_referencia, limite_parametros)
            
            parametros_batch.append((D, A, B, x, y))
            x, y = validate_ignition_point(x, y)
            parametros_validados.append((D, A, B, x, y, betas, gammas))
        
        # Calcular fitness en batch
        fitness_values = aptitud_batch(parametros_validados, celdas_quemadas_referencia, num_steps)
        
        # Agregar resultados
        for j, (params, fitness) in enumerate(zip(parametros_validados, fitness_values)):
            D, A, B, x, y, betas, gammas = params
            resultados.append({
                "D": D, "A": A, "B": B, "x": x, "y": y, "fitness": fitness,
                "betas": betas, "gammas": gammas
            })
            print(
                f'Individuo {i+j+1}: D={D:.4f}, A={A:.4f}, B={B:.4f}, x={x}, y={y}, '
                f'beta={[f"{b:.3f}" for b in betas]}, gamma={[f"{g:.3f}" for g in gammas]}, '
                f'fitness={fitness:.4f}'
            )
    
    return resultados

############################## ALGORITMO GENÉTICO #########################################################

def genetic_algorithm(tamano_poblacion, generaciones, limite_parametros, ruta_incendio_referencia, archivo_preentrenado=None, batch_size=10):
    """Implementa el algoritmo genético para estimar los parámetros del modelo de incendio."""
    
    # Obtener el task_id del SGE
    task_id = os.environ.get('JOB_ID', 'default')
    
    # Crear la carpeta de resultados con el task_id
    resultados_dir = f'resultados/task_{task_id}'
    os.makedirs(resultados_dir, exist_ok=True)

    # Cargar población inicial (preentrenada o nueva)
    if archivo_preentrenado:
        combinaciones = cargar_poblacion_preentrenada(archivo_preentrenado, tamano_poblacion, limite_parametros)
        
        # Verificar si los datos preentrenados incluyen fitness
        if len(combinaciones) > 0 and len(combinaciones[0]) == 6:  # D, A, B, x, y, fitness
            print("Datos preentrenados con fitness detectados. Convirtiendo a formato de resultados...")
            resultados = []
            for individuo in combinaciones:
                D, A, B, x, y, fitness = individuo
                resultados.append({
                    "D": float(D), "A": float(A), "B": float(B), 
                    "x": int(x), "y": int(y), "fitness": float(fitness)
                })
            print(f"Convertidos {len(resultados)} individuos preentrenados con fitness.")
        else:
            # Datos sin fitness, procesar normalmente
            print("Datos preentrenados sin fitness. Calculando fitness...")
            resultados = procesar_poblacion_batch(combinaciones, batch_size)
    else:
        combinaciones = poblacion_inicial(tamano_poblacion, limite_parametros)
        resultados = procesar_poblacion_batch(combinaciones, batch_size)
    
    mutation_rate = 0.3

    # Procesar población inicial en batch (solo si no tenemos datos preentrenados con fitness)
    if not archivo_preentrenado or len(combinaciones[0]) != 6:
        print("Procesando población inicial en batch...")
        resultados = procesar_poblacion_batch(combinaciones, ruta_incendio_referencia, limite_parametros, batch_size=batch_size)

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

        # Procesar nueva población en batch
        print(f"Procesando generación {gen+1} en batch...")
        resultados = procesar_poblacion_batch(population, ruta_incendio_referencia, limite_parametros, batch_size=batch_size)

        peor_idx = max(range(len(resultados)), key=lambda i: resultados[i]["fitness"])
        resultados[peor_idx] = elite  # Mantener el mejor individuo de la generación anterior

        best_fitness = min(resultados, key=lambda x: x["fitness"])["fitness"]
        print(f'Generación {gen+1}: Mejor fitness = {best_fitness}')

        # Opcional: reducir la tasa de mutación con el tiempo
        mutation_rate *= 0.99

        # Guardar los resultados de la generación en la carpeta específica del task_id
        csv_filename = os.path.join(resultados_dir, f'resultados_generacion_{gen+1}.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['D', 'A', 'B', 'x', 'y', 'fitness']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for resultado in resultados:
                writer.writerow(resultado)

    # Guardar resultados finales con información del task
    final_csv_filename = os.path.join(resultados_dir, f'resultados_finales_task_{task_id}.csv')
    with open(final_csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['D', 'A', 'B', 'x', 'y', 'fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for resultado in resultados:
            writer.writerow(resultado)
    
    print(f'Resultados guardados en: {resultados_dir}')
    print(f'Task ID: {task_id}')

    return resultados