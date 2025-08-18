from config import ruta_mapas
import numpy as np # type: ignore
import cupy as cp # type: ignore
import csv, os
from operadores_geneticos import poblacion_inicial

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
    # return cp.flipud(mapa_incendio_referencia)
    return mapa_incendio_referencia
    
############################## CALCULO A PARTIR DE MAPAS ###############################################

def preprocesar_datos():
    datos = [leer_asc(m) for m in ruta_mapas]
    vientod, vientov, pendiente, vegetacion, orientacion = datos

    # Parámetros derivados
    beta_veg = cp.where(vegetacion <= 2, 0, 0.1 * vegetacion)
    gamma = cp.where(vegetacion <= 2, 100, 0.1)

    vientod_rad = vientod * cp.pi / 180
    # Componentes cartesianas del viento:
    wx = -vientov * cp.sin(vientod_rad) * 1000  # Este = sin(ángulo desde Norte)
    wy = -vientov * cp.cos(vientod_rad) * 1000 # Norte = cos(ángulo desde Norte)

    h_dx = cp.tan(pendiente * cp.pi / 180) * cp.cos(orientacion * cp.pi / 180 - cp.pi/2)
    h_dy = cp.tan(pendiente * cp.pi / 180) * cp.sin(orientacion * cp.pi / 180 - cp.pi/2)

    return {
        "vientod": cp.flipud(vientod),
        "vientov": cp.flipud(vientov),
        "pendiente": cp.flipud(pendiente),
        "vegetacion": cp.flipud(vegetacion),
        "orientacion": cp.flipud(orientacion),
        "beta_veg": cp.flipud(beta_veg),
        "gamma": cp.flipud(gamma),
        "wx": cp.flipud(wx),
        "wy": cp.flipud(wy),
        "h_dx": cp.flipud(h_dx),
        "h_dy": cp.flipud(h_dy),
        "ny": vientod.shape[0],
        "nx": vientod.shape[1],
    }
############################## CARGA DE ARCHIVO PREENTRENADO ###############################################

def cargar_poblacion_preentrenada(archivo_preentrenado, tamano_poblacion, limite_parametros):
    """
    Carga una población preentrenada desde un CSV con columnas:
    D, A, B, x, y, beta_1..beta_5, gamma_1..gamma_5, fitness

    Devuelve una lista de individuos:
    [D, A, B, x, y, [beta1..beta5], [gamma1..gamma5], fitness]
    """

    if not os.path.exists(archivo_preentrenado):
        print(f"Archivo {archivo_preentrenado} no encontrado. Generando población inicial.")
        return poblacion_inicial(tamano_poblacion, limite_parametros)

    poblacion_cargada = []
    with open(archivo_preentrenado, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Primeros 5 parámetros
                parametros_basicos = [
                    float(row['D']),
                    float(row['A']),
                    float(row['B']),
                    int(float(row['x'])),
                    int(float(row['y']))
                ]

                # Betas y gammas
                betas = [float(row[f'beta_{i}']) for i in range(1, 6)]
                gammas = [float(row[f'gamma_{i}']) for i in range(1, 6)]

                # Fitness
                fitness = float(row['fitness'])

                # Guardar como lista de 8 elementos
                poblacion_cargada.append(parametros_basicos + [betas, gammas, fitness])

            except (ValueError, KeyError) as e:
                print(f"WARNING: Saltando fila inválida: {row} - Error: {e}")
                continue

    # Ajustar tamaño de la población
    num_cargados = len(poblacion_cargada)
    if num_cargados == 0:
        print("No se cargaron individuos válidos. Generando población inicial.")
        return poblacion_inicial(tamano_poblacion, limite_parametros)

    if num_cargados > tamano_poblacion:
        indices = cp.random.choice(num_cargados, tamano_poblacion, replace=False)
        poblacion_cargada = [poblacion_cargada[i] for i in indices.get()]
    elif num_cargados < tamano_poblacion:
        faltantes = tamano_poblacion - num_cargados
        nuevos = poblacion_inicial(faltantes, limite_parametros)
        # Cada nuevo individuo se convierte en 5 parámetros + listas vacías + fitness None
        nuevos_formateados = [n + [[], [], None] for n in nuevos]
        poblacion_cargada += nuevos_formateados

    return poblacion_cargada

    
############################## GUARDADO DE RESULTADOS ###############################################

def guardar_resultados(resultados, resultados_dir, gen, n_betas=5, n_gammas=5):
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
    fieldnames = ['D', 'A', 'B', 'x', 'y'] \
               + [f'beta_{i}' for i in range(1, n_betas+1)] \
               + [f'gamma_{i}' for i in range(1, n_gammas+1)] \
               + ['fitness']
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for resultado in resultados:
            row = {
                'D': resultado['D'],
                'A': resultado['A'],
                'B': resultado['B'],
                'x': resultado['x'],
                'y': resultado['y'],
                'fitness': resultado['fitness'],
            }
            
            # Expandir betas
            for i, beta in enumerate(resultado['betas'], start=1):
                row[f'beta_{i}'] = beta
            
            # Expandir gammas
            for i, gamma in enumerate(resultado['gammas'], start=1):
                row[f'gamma_{i}'] = gamma
            
            writer.writerow(row)
    
    print(f"✅ Resultados guardados en {csv_filename}")