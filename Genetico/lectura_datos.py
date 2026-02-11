import numpy as np # type: ignore
import cupy as cp # type: ignore
import csv, os
from population import Population

############################## LEER MAPAS RASTER ###############################################

def leer_asc(ruta):
    with open(ruta, 'r') as f:
        for _ in range(6):  
            f.readline()
        data = np.loadtxt(f) 
        return cp.array(data, dtype=cp.float32) 
    
############################## LEER MAPA INCENDIO DE REFERENCIA ###############################################

def leer_incendio_referencia(ruta):
    _, extension = os.path.splitext(ruta)
    if extension == '.asc':
        mapa_incendio_referencia = leer_asc(ruta)
    elif extension == '.npy':
        mapa_incendio_referencia = cp.load(ruta)
        if mapa_incendio_referencia.ndim == 3:
            mapa_incendio_referencia = mapa_incendio_referencia[0]
    else:
        raise ValueError(f'Extensión no reconocida: {extension}')
    return mapa_incendio_referencia
    
############################## CARGA DE ARCHIVO PREENTRENADO ###############################################

def cargar_poblacion_preentrenada(archivo_preentrenado, tamano_poblacion, limite_parametros, 
                                  n_betas=5, n_gammas=5,
                                  ajustar_beta_gamma=True, ajustar_ignicion=True):
    """
    Carga una población preentrenada desde un CSV con columnas:
    D, A, B, x, y, beta_1..beta_5, gamma_1..gamma_5, fitness

    Devuelve: lista de diccionarios con claves
    ['D','A','B','x','y','betas','gammas','fitness'].
    'betas' y 'gammas' son cp.ndarray (float32).
    """

    print(f"[DEBUG] Cargando población preentrenada desde: {archivo_preentrenado}")

    if not os.path.exists(archivo_preentrenado):
        raise ValueError(f"[DEBUG] Archivo {archivo_preentrenado} no encontrado.")

    poblacion_cargada = []
    with open(archivo_preentrenado, 'r') as f:
        reader = csv.DictReader(f)
        total_rows = sum(1 for _ in open(archivo_preentrenado)) - 1  # sin header
        f.seek(0)  # volver al inicio después del conteo

        for idx, row in enumerate(reader, start=1):
            try:
                D = float(row['D']); A = float(row['A']); B = float(row['B'])

                if ajustar_ignicion:
                    x = int(float(row['x'])); y = int(float(row['y']))

                if ajustar_beta_gamma and 'beta_1' in row: # Exp3
                    betas = cp.array([float(row[f'beta_{i}']) for i in range(1, n_betas+1)],
                                      dtype=cp.float32)
                    gammas = cp.array([float(row[f'gamma_{i}']) for i in range(1, n_gammas+1)],
                                       dtype=cp.float32)
                elif ajustar_beta_gamma and 'beta' in row: # Exp2
                    betas = cp.array(float(row['beta']), dtype=cp.float32)
                    gammas = cp.array(float(row['gamma']), dtype=cp.float32)

                fval = row.get('fitness', '')
                fitness = (float(fval) if (fval is not None and fval != '') else None)

                if ajustar_beta_gamma and ajustar_ignicion:  # Exp2
                    poblacion_cargada.append({
                        "D": D, "A": A, "B": B, "x": x, "y": y,
                        "betas": betas, "gammas": gammas, "fitness": fitness
                    })
                    # Prints de debug solo en casos selectos
                    if idx <= 3 or idx == total_rows // 2 or idx == total_rows:
                        print(f"[DEBUG] Fila {idx}/{total_rows}: "
                            f"D={D}, A={A}, B={B}, x={x}, y={y}, "
                            f"betas={betas.get()}, gammas={gammas.get()}, fitness={fitness}")
                elif ajustar_beta_gamma and not ajustar_ignicion:    # Exp3
                    poblacion_cargada.append({
                        "D": D, "A": A, "B": B,
                        "betas": betas, "gammas": gammas, "fitness": fitness
                    })
                    # Prints de debug solo en casos selectos
                    if idx <= 3 or idx == total_rows // 2 or idx == total_rows:
                        print(f"[DEBUG] Fila {idx}/{total_rows}: "
                            f"D={D}, A={A}, B={B}, "
                            f"betas={betas.get()}, gammas={gammas.get()}, fitness={fitness}")
                else:                                        # Exp1
                    poblacion_cargada.append({
                        "D": D, "A": A, "B": B, "x": x, "y": y, "fitness": fitness 
                    })
                    # Prints de debug solo en casos selectos
                    if idx <= 3 or idx == total_rows // 2 or idx == total_rows:
                        print(f"[DEBUG] Fila {idx}/{total_rows}: "
                            f"D={D}, A={A}, B={B}, x={x}, y={y}, fitness={fitness}")

            except (ValueError, KeyError) as e:
                if idx <= 5:  # solo aviso explícito en las primeras filas
                    print(f"[WARNING] Fila inválida {idx}, se salta. Error: {e}")
                continue

    # Ajustar tamaño de la población
    num_cargados = len(poblacion_cargada)
    print(f"\n[DEBUG] Total individuos cargados: {num_cargados}")

    if num_cargados == 0:
        raise ValueError("[DEBUG] No se cargaron individuos válidos.")

    if num_cargados > tamano_poblacion:
        print(f"[DEBUG] Se cargaron {num_cargados}, recortando a {tamano_poblacion}")
        indices = cp.random.choice(num_cargados, tamano_poblacion, replace=False)
        poblacion_cargada = [poblacion_cargada[i] for i in indices.get()]
    elif num_cargados < tamano_poblacion:
        faltantes = tamano_poblacion - num_cargados
        print(f"[DEBUG] Faltan {faltantes} individuos. Generando población inicial para completar.")
        nuevos_pop = Population.initial_population(faltantes, limite_parametros)

        # Reformatear cada nuevo individuo con las mismas claves que los cargados
        for ind in nuevos_pop.individuals:
            ind_dict = ind.as_dict(ajustar_beta_gamma, ajustar_ignicion)
            poblacion_cargada.append(ind_dict)

    print(f"[DEBUG] Población final: {len(poblacion_cargada)} individuos.")
    return Population.from_results(poblacion_cargada, ajustar_beta_gamma, ajustar_ignicion)

############################## GUARDADO DE RESULTADOS ###############################################

def guardar_resultados(poblacion, resultados_dir, gen, n_betas=5, n_gammas=5, 
                       ajustar_beta_gamma=True, ajustar_ignicion=True):
    """
    Guarda resultados en un archivo CSV.
    
    resultados: lista de diccionarios con claves 
                ['D', 'A', 'B', 'x', 'y', 'betas', 'gammas', 'fitness']
    resultados_dir: carpeta donde guardar
    gen: número de generación (se incluye en el nombre del archivo)
    n_betas: cantidad de betas esperadas por fila   
    n_gammas: cantidad de gammas esperadas por fila
    """
    
    csv_filename = os.path.join(resultados_dir, f'resultados_generacion_{gen+1}.csv')
    
    # Definir nombres de columnas dinámicamente
    if ajustar_beta_gamma and ajustar_ignicion:   # Exp2
        fieldnames = ['D', 'A', 'B', 'x', 'y'] \
                   + ['beta'] \
                   + ['gamma'] \
                   + ['fitness']
    elif ajustar_beta_gamma and not ajustar_ignicion:  # Exp3
        fieldnames = ['D', 'A', 'B'] \
                   + [f'beta_{i}' for i in range(1, n_betas+1)] \
                   + [f'gamma_{i}' for i in range(1, n_gammas+1)] \
                   + ['fitness']
    else:                                                      # Exp1
        fieldnames = ['D', 'A', 'B', 'x', 'y'] + ['fitness'] 

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for individuo in poblacion.individuals:
            resultado = individuo.as_dict(ajustar_beta_gamma, ajustar_ignicion)
            row = {
                'D': resultado['D'],
                'A': resultado['A'],
                'B': resultado['B'],
                'fitness': resultado['fitness'],
            }

            if ajustar_ignicion:
                row['x'] = resultado['x']
                row['y'] = resultado['y']
            
            if ajustar_beta_gamma:
                betas = resultado['betas']
                gammas = resultado['gammas']

                # Convertir a array siempre
                if not isinstance(betas, (cp.ndarray, np.ndarray)):
                    betas = cp.array([betas], dtype=cp.float32)
                if not isinstance(gammas, (cp.ndarray, np.ndarray)):
                    gammas = cp.array([gammas], dtype=cp.float32)

                if betas.size > 1:
                    # Expandir betas
                    for i, beta in enumerate(betas, start=1):
                        row[f'beta_{i}'] = float(beta)  # aseguro que sea serializable
                    # Expandir gammas
                    for i, gamma in enumerate(gammas, start=1):
                        row[f'gamma_{i}'] = float(gamma)
                else:
                    row['beta'] = float(betas)
                    row['gamma'] = float(gammas)
            writer.writerow(row)
    
    print(f"✅ Resultados guardados en {csv_filename}")