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
    
############################## CALCULO A PARTIR DE MAPAS ###############################################

def preprocesar_datos():
    datos = [leer_asc(m) for m in ruta_mapas]
    vientod, vientov, pendiente, vegetacion, orientacion, area_quemada = datos

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
        "area_quemada": cp.flipud(area_quemada),
        "ny": vientod.shape[0],
        "nx": vientod.shape[1],
    }

############################## CARGA DE ARCHIVO PREENTRENADO ###############################################

def cargar_poblacion_preentrenada(archivo_preentrenado, tamano_poblacion, limite_parametros):
    """
    Carga una población preentrenada desde un archivo CSV.
    
    Args:
        archivo_preentrenado: Ruta al archivo CSV con individuos preentrenados
        tamano_poblacion: Tamaño deseado de la población
        limite_parametros: Límites para generar individuos adicionales si es necesario
    
    Returns:
        Lista de individuos (arrays de CuPy)
    """
    print(f'Cargando archivo preentrenado: {archivo_preentrenado}')
    
    # Verificar que el archivo existe
    if not os.path.exists(archivo_preentrenado):
        print(f'ERROR: Archivo {archivo_preentrenado} no encontrado. Generando población inicial.')
        return poblacion_inicial(tamano_poblacion, limite_parametros)
    
    try:
        with open(archivo_preentrenado, 'r') as f:
            reader = csv.DictReader(f)
            combinaciones_preentrenadas = []
            
            for row in reader:
                # Validar que todas las columnas necesarias existen
                if all(key in row for key in ['D', 'A', 'B', 'x', 'y']):
                    try:
                        # Convertir a float primero y luego a int para manejar valores como "732.0"
                        x_val = int(float(row['x']))
                        y_val = int(float(row['y']))
                        
                        # Debug temporal: verificar los primeros valores
                        if len(combinaciones_preentrenadas) < 3:
                            print(f"DEBUG: Cargando fila {len(combinaciones_preentrenadas)+1}: x={row['x']}→{x_val}, y={row['y']}→{y_val}")
                        
                        combinaciones_preentrenadas.append(
                            cp.array([float(row['D']), float(row['A']), float(row['B']), 
                                     x_val, y_val])
                        )
                    except (ValueError, TypeError) as e:
                        print(f"WARNING: Saltando fila inválida: {row} - Error: {e}")
                        continue
        
        # Verificar que se cargaron individuos
        if not combinaciones_preentrenadas:
            print('WARNING: No se encontraron individuos válidos en el archivo. Generando población inicial.')
            return poblacion_inicial(tamano_poblacion, limite_parametros)
        
        # Ajustar el tamaño de la población
        num_cargados = len(combinaciones_preentrenadas)
        print(f'Cargados {num_cargados} individuos del archivo preentrenado.')
        
        if num_cargados == tamano_poblacion:
            # Perfecto, usar todos
            return combinaciones_preentrenadas
        elif num_cargados > tamano_poblacion:
            # Tomar una muestra aleatoria
            indices = cp.random.choice(num_cargados, tamano_poblacion, replace=False)
            combinaciones = [combinaciones_preentrenadas[i] for i in indices.get()]
            print(f'Tomando una muestra de {tamano_poblacion} individuos del archivo.')
            return combinaciones
        else:
            # Completar con individuos generados aleatoriamente
            faltantes = tamano_poblacion - num_cargados
            nuevos = poblacion_inicial(faltantes, limite_parametros)
            combinaciones = combinaciones_preentrenadas + nuevos
            print(f'Completando con {faltantes} individuos generados aleatoriamente.')
            return combinaciones
            
    except Exception as e:
        print(f'ERROR al cargar archivo preentrenado: {e}. Generando población inicial.')
        return poblacion_inicial(tamano_poblacion, limite_parametros)