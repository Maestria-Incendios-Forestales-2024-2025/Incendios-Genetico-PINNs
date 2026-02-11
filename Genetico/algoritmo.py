import cupy as cp  # type: ignore
import os
from operadores_geneticos import TournamentSelection, OnePointCrossover, GaussianMutation
from fitness import FitnessEvaluator
from population import Population
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
        Población con fitness calculado
    """  
    evaluator = FitnessEvaluator(ctx)

    incendio_referencia = leer_incendio_referencia(ruta_incendio_referencia)
    celdas_quemadas_referencia = cp.where(incendio_referencia > 0.001, 1, 0)

    population_fitness = []

    for i in range(0, len(poblacion), batch_size):
        # Accede a los individuos del batch
        batch = poblacion.individuals[i:i+batch_size]
        print(f'Procesando batch {i//batch_size + 1} de {len(poblacion) // batch_size}...')
        # Accede a los parámetros de los individuos del batch
        parametros_batch = [individuo.genes for individuo in batch]

        # Validación de parámetros
        parametros_validados = []
        if ajustar_beta_gamma and ajustar_ignicion:             # Exp2
            for D, A, B, x, y, betas, gammas in parametros_batch:
                A, B = evaluator.validate_courant_and_adjust(A, B)
                x, y = evaluator.validate_ignition_point(x, y, incendio_referencia, limite_parametros)
                betas, gammas = evaluator.validate_beta_gamma(betas, gammas)
                parametros_validados.append((D, A, B, x, y, betas, gammas))
        elif ajustar_beta_gamma and not ajustar_ignicion:       # Exp3 
            for genes in parametros_batch:
                D, A, B = genes[0], genes[1], genes[2]
                betas = genes[3:8]   # índices 3, 4, 5, 6, 7
                gammas = genes[8:13] # índices 8, 9, 10, 11, 12
                A, B = evaluator.validate_courant_and_adjust(A, B)
                betas, gammas = evaluator.validate_beta_gamma(betas, gammas)
                parametros_validados.append((D, A, B, betas, gammas))
        else:                                                   # Exp1
            for D, A, B, x, y in parametros_batch:
                A, B = evaluator.validate_courant_and_adjust(A, B)
                x, y = evaluator.validate_ignition_point(x, y, incendio_referencia, limite_parametros)
                parametros_validados.append((D, A, B, x, y))

        # Lista de valores de fitness 
        fitness_values = evaluator.evaluate_batch(parametros_validados, celdas_quemadas_referencia, num_steps, 
                                       ajustar_beta_gamma=ajustar_beta_gamma, beta_fijo=beta_fijo, gamma_fijo=gamma_fijo, 
                                       ajustar_ignicion=ajustar_ignicion, ignicion_fija_x=ignicion_fija_x, 
                                       ignicion_fija_y=ignicion_fija_y)

        population_fitness.extend(fitness_values)

    for individuo, fitness in zip(poblacion.individuals, population_fitness):
        individuo.update_fitness(fitness)

    return poblacion

class GeneticAlgorithm:
    def __init__(self, tamano_poblacion, generaciones, limite_parametros, ruta_incendio_referencia, ctx,
                 archivo_preentrenado = None, generacion_preentrenada=0, num_steps=10000, batch_size=10,
                 ajustar_beta_gamma=True, beta_fijo=None, gamma_fijo=None, ajustar_ignicion=True,
                 ignicion_fija_x=None, ignicion_fija_y=None):
        
        self.tamano_poblacion = tamano_poblacion
        self.generaciones = generaciones
        self.limite_parametros = limite_parametros
        self.ruta_incendio_referencia = ruta_incendio_referencia
        self.ctx = ctx
        self.archivo_preentrenado = archivo_preentrenado
        self.generacion_preentrenada = generacion_preentrenada
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.ajustar_beta_gamma=ajustar_beta_gamma
        self.beta_fijo=beta_fijo
        self.gamma_fijo=gamma_fijo
        self.ajustar_ignicion=ajustar_ignicion
        self.ignicion_fija_x=ignicion_fija_x
        self.ignicion_fija_y=ignicion_fija_y

        # Operadores
        self.selection_op = TournamentSelection()
        self.crossover_op = OnePointCrossover()
        self.mutation_op = GaussianMutation()
    
    def initialize(self):
        if self.archivo_preentrenado: # Si hay una población preentrenada la carga
            poblacion = cargar_poblacion_preentrenada(
                self.archivo_preentrenado, 
                self.tamano_poblacion, 
                self.limite_parametros,
                ajustar_beta_gamma=self.ajustar_beta_gamma, 
                ajustar_ignicion=self.ajustar_ignicion
            )
        else: # Instancio la población inicial
            poblacion = Population.initial_population(self.tamano_poblacion, self.limite_parametros)
            poblacion = self.evaluate_population(poblacion)
        
        return poblacion
    
    def evaluate_population(self, poblacion):
        return procesar_poblacion_batch(
            poblacion,
            self.ruta_incendio_referencia,
            self.limite_parametros,
            self.ctx,
            num_steps=self.num_steps,
            batch_size=self.batch_size,
            ajustar_beta_gamma=self.ajustar_beta_gamma,
            beta_fijo=self.beta_fijo,
            gamma_fijo=self.gamma_fijo,
            ajustar_ignicion=self.ajustar_ignicion,
            ignicion_fija_x=self.ignicion_fija_x,
            ignicion_fija_y=self.ignicion_fija_y
        )
    
    def produce_offspring(self, poblacion, mutation_rate):
        new_population = []
        for _ in range(self.tamano_poblacion // 2):
            parent1 = self.selection_op.select(poblacion)
            parent2 = self.selection_op.select(poblacion)
            child1, child2 = self.crossover_op.apply(parent1, parent2)
            child1 = self.mutation_op.mutate(child1, mutation_rate, self.limite_parametros)
            new_population.extend([child1, child2])
        return Population(new_population, generation=poblacion.generation + 1)
    
    def apply_elitism(self, poblacion, elite):
        peor = poblacion.worst()
        elite_clonada = elite.clone()
        peor_idx = poblacion.individuals.index(peor)
        poblacion.individuals[peor_idx] = elite_clonada
        return poblacion
    
    def step(self, poblacion, mutation_rate, gen):
        elite = poblacion.best()
        nueva = self.produce_offspring(poblacion, mutation_rate)
        nueva = self.evaluate_population(nueva)
        nueva = self.apply_elitism(nueva, elite)
        return nueva
    
    def run(self):
        resultados_dir = "resultados"
        os.makedirs(resultados_dir, exist_ok=True)

        poblacion = self.initialize()
        mutation_rate = 0.3 * 0.99**self.generacion_preentrenada

        for gen in range(self.generaciones + 1):
            if gen > 0:
                print(f"Iniciando generación {gen}...")
                poblacion = self.step(poblacion, mutation_rate, gen)
                mutation_rate *= 0.99
            
            print(f"Generación {gen}: Mejor fitness = {poblacion.best().fitness}")

            for i, ind in enumerate(poblacion.individuals, 1):
                print(f"Individuo {i}: {ind.as_dict(self.ajustar_beta_gamma, self.ajustar_ignicion)}")
            
            guardar_resultados(poblacion, resultados_dir, gen - 1 + self.generacion_preentrenada,
                               ajustar_beta_gamma=self.ajustar_beta_gamma, 
                               ajustar_ignicion=self.ajustar_ignicion)
            
        print(f"Resultados guardados en: {resultados_dir}")
        return poblacion

############################## ALGORITMO GENÉTICO #########################################################

def genetic_algorithm(tamano_poblacion, generaciones, limite_parametros, ruta_incendio_referencia, ctx,
                 archivo_preentrenado = None, generacion_preentrenada=0, num_steps=10000, batch_size=10,
                 ajustar_beta_gamma=True, beta_fijo=None, gamma_fijo=None, ajustar_ignicion=True,
                 ignicion_fija_x=None, ignicion_fija_y=None):
    
    ga = GeneticAlgorithm(tamano_poblacion, generaciones, limite_parametros, ruta_incendio_referencia, ctx,
                 archivo_preentrenado, generacion_preentrenada, num_steps, batch_size,
                 ajustar_beta_gamma, beta_fijo, gamma_fijo, ajustar_ignicion,
                 ignicion_fija_x, ignicion_fija_y)
    
    return ga.run()