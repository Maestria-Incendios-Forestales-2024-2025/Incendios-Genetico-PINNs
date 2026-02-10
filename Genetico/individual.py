import cupy as cp

class Individual:
    def __init__(self, genes, fitness=None):
        self.genes = genes
        self.fitness = fitness

    def clone(self):
        genes_copy = cp.copy(self.genes) if hasattr(cp, "array") else self.genes[:]
        return Individual(genes_copy, self.fitness)

    def invalidate_fitness(self):
        self.fitness = None
    
    def __len__(self):
        return len(self.genes)
    
    def to_array(self):
        return cp.asarray(self.genes)

    def as_dict(self, ajustar_beta_gamma=True, ajustar_ignicion=True):
        # Mapea genes
        D, A, B = self.genes[:3]

        result = {"D": float(D), "A": float(A), "B": float(B), "fitness": self.fitness}

        if ajustar_beta_gamma and ajustar_ignicion: # Exp2
            x, y = self.genes[3], self.genes[4]
            betas = self.genes[5]
            gammas = self.genes[6]
            result.update({"x": int(x), "y": int(y), "betas": betas, "gammas": gammas})

        elif ajustar_beta_gamma and not ajustar_ignicion: # Exp3
            betas = self.genes[3:8]
            gammas = self.genes[8:13]
            result.update({"betas": betas, "gammas": gammas})
        
        else: # Exp1
            x, y = self.genes[3], self.genes[4]
            result.update({"x": int(x), "y": int(y)})

        return result

    