import cupy as cp  # type: ignore
import csv
import sys
import os
from operadores_geneticos import poblacion_inicial, tournament_selection, crossover, mutate
from fitness import aptitud_batch
from config import d, dt
from lectura_datos import preprocesar_datos, cargar_poblacion_preentrenada

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
area_quemada = datos["area_quemada"]

############################## CHEQUEO DE CONDICIÓN DE COURANT ###############################################

def validate_courant_and_adjust(D, A, B):
    """Valida la condición de Courant y ajusta parámetros si es necesario."""
    while not courant(dt, D, A, B, d, wx, wy, h_dx=h_dx_mapa, h_dy=h_dy_mapa):
        param_to_modify = cp.random.choice(["D", "A", "B"])
        
        if param_to_modify == "D":
            D = float(D * float(cp.random.uniform(0.8, 0.99)))
        elif param_to_modify == "A":
            A = float(A * float(cp.random.uniform(0.8, 0.99)))
        elif param_to_modify == "B":
            B = float(B * float(cp.random.uniform(0.8, 0.99)))
    
    return D, A, B

############################## PROCESAMIENTO EN BATCH ###############################################

def procesar_poblacion_batch(poblacion, batch_size=10):
    """
    Procesa una población en batches para aprovechar el paralelismo.
    
    Args:
        poblacion: Lista de individuos (D, A, B, x, y)
        batch_size: Tamaño del batch para procesamiento en paralelo
    
    Returns:
        Lista de resultados con fitness calculado
    """
    resultados = []
    
    for i in range(0, len(poblacion), batch_size):
        batch = poblacion[i:i+batch_size]
        
        # Preparar parámetros para el batch
        parametros_batch = []
        for individuo in batch:
            D, A, B, x, y = individuo
            D, A, B, x, y = D.item(), A.item(), B.item(), int(x.item()), int(y.item())
            
            # Validar y ajustar parámetros
            D, A, B = validate_courant_and_adjust(D, A, B)
            x, y = validate_ignition_point(x, y)
            
            parametros_batch.append((D, A, B, x, y))
        
        # Calcular fitness en batch
        fitness_values = aptitud_batch(parametros_batch)
        
        # Agregar resultados
        for j, (params, fitness) in enumerate(zip(parametros_batch, fitness_values)):
            D, A, B, x, y = params
            resultados.append({
                "D": D, "A": A, "B": B, "x": x, "y": y, "fitness": fitness
            })
            print(f'Individuo {i+j+1}: D={D:.4f}, A={A:.4f}, B={B:.4f}, x={x}, y={y}, fitness={fitness:.4f}')
    
    return resultados

############################## VALIDACIÓN DE PUNTO DE IGNICIÓN ###############################################

def validate_ignition_point(x, y):
    """Valida que el punto de ignición tenga combustible."""
    while vegetacion[int(y), int(x)] <= 2 and area_quemada[int(y), int(x)] <= 0.001:
        x, y = float(cp.random.randint(300, 720)), float(cp.random.randint(400, 800))
    return x, y

############################## ALGORITMO GENÉTICO #########################################################

def genetic_algorithm(tamano_poblacion, generaciones, limite_parametros, archivo_preentrenado=None, batch_size=10):
    """Implementa el algoritmo genético para estimar los parámetros del modelo de incendio."""
    
    # Obtener el task_id del SGE
    task_id = os.environ.get('JOB_ID', 'default')
    
    # Crear la carpeta de resultados con el task_id
    resultados_dir = f'Genetico/resultados/task_{task_id}'
    os.makedirs(resultados_dir, exist_ok=True)
    
    # Cargar población inicial (preentrenada o nueva)
    if archivo_preentrenado:
        combinaciones = cargar_poblacion_preentrenada(archivo_preentrenado, tamano_poblacion, limite_parametros)
    else:
        combinaciones = poblacion_inicial(tamano_poblacion, limite_parametros)

    mutation_rate = 0.3

    # Procesar población inicial en batch
    print("Procesando población inicial en batch...")
    resultados = procesar_poblacion_batch(combinaciones, batch_size)

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
        resultados = procesar_poblacion_batch(population, batch_size)

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