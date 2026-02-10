import cupy as cp
from individual import Individual

class Population:
    def __init__(self, individuals, generation=0):
        self.individuals = individuals
        self.generation = generation
    
    @classmethod
    def initial_population(cls, size, limite_parametros):
        """
        Genera población inicial dentro de los límites establecidos
        Args:
            size: Número de individuos
            limite_parametros: Lista de tuplas (min, max) para cada parámetro
        Returns:
            Lista de individuos (cada individuo es una instancia de Individual)
        """
        n_params = len(limite_parametros)
    
        # Genera números aleatorios uniformes en [0,1]
        rand = cp.random.rand(size, n_params, dtype=cp.float32)
    
        # Convierte límites a arrays
        lows  = cp.array([low for low, _ in limite_parametros], dtype=cp.float32)
        highs = cp.array([high for _, high in limite_parametros], dtype=cp.float32)

        # Escala cada columna al rango correspondiente
        genes = lows + rand * (highs - lows)
        individuals = [Individual(genes[i]) for i in range(size)]

        return cls(individuals, generation=0)
    
    def best(self):
        return min([ind for ind in self.individuals if ind.fitness is not None], key=lambda ind: ind.fitness)
    
    def worst(self):
        return max([ind for ind in self.individuals if ind.fitness is not None], key=lambda ind: ind.fitness)
    
    def sort_by_fitness(self):
        self.individuals.sort(key=lambda ind: ind.fitness)
        
    def to_array(self):
        return cp.asarray([ind.genes for ind in self.individuals])
    
    def replace_worst(self, individual):
        self.sort_by_fitness()
        self.individuals[-1] = individual 

    def mean_fitness(self):
        fitness_values = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        if not fitness_values:
            return None
        return sum(fitness_values) / len(fitness_values)

    @classmethod
    def from_results(cls, resultados, ajustar_beta_gamma, ajustar_ignicion):
        individuals = []
        for res in resultados:
            D, A, B = res['D'], res['A'], res['B']
            fitness = res['fitness']

            if ajustar_beta_gamma and ajustar_ignicion: # Exp2
                x, y = res['x'], res['y']
                betas, gammas = res['betas'], res['gammas']
                genes = cp.array([D, A, B, betas, gammas, x, y])
            
            elif ajustar_beta_gamma and not ajustar_ignicion: # Exp3
                betas, gammas = res['betas'], res['gammas']
                genes = cp.array([D, A, B, betas, gammas])
        
            else: # Exp1
                x, y = res['x'], res['y']
                genes = cp.array([D, A, B, x, y])
        
            individuals.append(Individual(genes, fitness))

        return cls(individuals)