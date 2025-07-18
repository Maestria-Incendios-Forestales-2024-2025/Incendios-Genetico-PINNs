import cupy as cp # type: ignore
from config import d, dt, num_steps
from lectura_datos import preprocesar_datos
import sys
import os

# Agrega el directorio padre al path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelo_rdc import spread_infection_raw

############################## FUNCIÓN PARA AGREGAR UNA DIMENSIÓN ###############################################

def ensure_batch_dim(*arrays):
    return [arr if arr.ndim == 3 else arr[cp.newaxis, ...] for arr in arrays]

############################## UTILIDADES PARA PARÁMETROS DE VEGETACIÓN ###############################################

def get_vegetation_info(vegetacion):
    """
    Obtiene información sobre los tipos de vegetación en el mapa.
    
    Returns:
        dict: Información sobre tipos de vegetación
    """
    veg_types = cp.unique(vegetacion)
    veg_counts = {int(vt): int(cp.sum(vegetacion == vt)) for vt in veg_types}
    
    print("Información de vegetación:")
    for veg_type, count in veg_counts.items():
        percentage = (count / (ny * nx)) * 100
        print(f"  Tipo {veg_type}: {count} celdas ({percentage:.1f}%)")
    
    return {
        "types": [int(vt) for vt in veg_types],
        "counts": veg_counts,
        "total_types": len(veg_types)
    }

############################## CARGADO DE MAPAS ###############################################

datos = preprocesar_datos()

vegetacion = datos["vegetacion"]
wx = datos["wx"]
wy = datos["wy"]
h_dx_mapa = datos["h_dx"]
h_dy_mapa = datos["h_dy"]
area_quemada = datos["area_quemada"]
ny, nx = datos["ny"], datos["nx"]
# Obtener información de vegetación al inicializar
veg_info = get_vegetation_info(vegetacion)

############################## INCENDIO DE REFERENCIA ###############################################

area_quemada = datos["area_quemada"]
burnt_cells = cp.where(area_quemada > 0.001, 1, 0)  # Celdas quemadas en el mapa de referencia

############################## FUNCIÓN PARA MAPEAR PARÁMETROS DE VEGETACIÓN ###############################################

def crear_mapas_parametros_batch(parametros_batch, vegetacion):
    """
    Crea mapas de beta y gamma personalizados para cada simulación en el batch
    basados en los parámetros optimizados y el tipo de vegetación.
    
    Args:
        parametros_batch: Lista de tuplas (D, A, B, x, y, beta_params, gamma_params)
        vegetacion: Mapa de tipos de vegetación (ny, nx)
    
    Returns:
        beta_batch, gamma_batch: Arrays (batch_size, ny, nx) con valores mapeados
    """
    batch_size = len(parametros_batch)
    
    # Crear arrays de salida
    beta_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
    gamma_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
    
    # Obtener tipos únicos de vegetación
    veg_types = cp.array([3, 4, 5, 6, 7], dtype=cp.int32)

    print(f"Tipos de vegetación únicos: {veg_types}")
    
    # Para cada simulación en el batch
    for i, params in enumerate(parametros_batch):
        if len(params) >= 7:  # Verificar que tenemos los parámetros de vegetación
            beta_params = params[5]  # Lista de betas por tipo de vegetación
            gamma_params = params[6]  # Lista de gammas por tipo de vegetación
            
            # Inicializar con valores por defecto
            beta_map = cp.zeros_like(vegetacion, dtype=cp.float32)
            gamma_map = cp.zeros_like(vegetacion, dtype=cp.float32)
            
            # Mapear valores según tipo de vegetación
            for j, veg_type in enumerate(veg_types):
                veg_type = int(veg_type)
                # Crear máscara para este tipo de vegetación
                mask = (vegetacion == veg_type)
                
                # Asignar valores si tenemos parámetros para este tipo
                if j < len(beta_params) and j < len(gamma_params):
                    beta_map = cp.where(mask, beta_params[j], beta_map)
                    gamma_map = cp.where(mask, gamma_params[j], gamma_map)
            
            beta_batch[i] = beta_map
            gamma_batch[i] = gamma_map
    
    return beta_batch, gamma_batch

############################## CÁLCULO DE FUNCIÓN DE FITNESS POR BATCH ###############################################

def aptitud_batch(parametros_batch):
    """
    Calcula el fitness para múltiples combinaciones de parámetros en paralelo.
    
    Args:
        parametros_batch: Lista de tuplas (D, A, B, x, y, beta_params, gamma_params)
    
    Returns:
        Lista de valores de fitness
    """
    batch_size = len(parametros_batch)
    
    # Inicializar arrays para el batch
    S_batch = cp.ones((batch_size, ny, nx), dtype=cp.float32)
    I_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
    R_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
    
    # Configurar puntos de ignición para cada simulación
    for i, params in enumerate(parametros_batch):
        # Extraer coordenadas (D, A, B, x, y, ...)
        x, y = int(params[3]), int(params[4])
        S_batch[i, y, x] = 0
        I_batch[i, y, x] = 1
    
    # Arrays para los nuevos estados
    S_new_batch = cp.empty_like(S_batch)
    I_new_batch = cp.empty_like(I_batch)
    R_new_batch = cp.empty_like(R_batch)
    
    # Crear mapas de parámetros de vegetación personalizados
    beta_batch, gamma_batch = crear_mapas_parametros_batch(parametros_batch, vegetacion)

    # Expandir arrays de parámetros ambientales para el batch
    wx_batch = cp.broadcast_to(wx, (batch_size, ny, nx))
    wy_batch = cp.broadcast_to(wy, (batch_size, ny, nx))
    h_dx_batch = cp.broadcast_to(h_dx_mapa, (batch_size, ny, nx))
    h_dy_batch = cp.broadcast_to(h_dy_mapa, (batch_size, ny, nx))
    
    # Crear arrays de parámetros D, A, B para cada simulación
    D_batch = cp.array([param[0] for param in parametros_batch], dtype=cp.float32)
    A_batch = cp.array([param[1] for param in parametros_batch], dtype=cp.float32)
    B_batch = cp.array([param[2] for param in parametros_batch], dtype=cp.float32)

    # Simular en paralelo
    simulaciones_validas = cp.ones(batch_size, dtype=cp.bool_)
    
    for t in range(num_steps):
        # Llamar al kernel con todos los parámetros necesarios
        spread_infection_raw(
            S=S_batch, I=I_batch, R=R_batch, 
            S_new=S_new_batch, I_new=I_new_batch, R_new=R_new_batch,
            dt=dt, d=d, beta=beta_batch, gamma=gamma_batch,
            D=D_batch, wx=wx_batch, wy=wy_batch, 
            h_dx=h_dx_batch, h_dy=h_dy_batch, A=A_batch, B=B_batch
        )
        
        # Intercambiar arrays
        S_batch, S_new_batch = S_new_batch, S_batch
        I_batch, I_new_batch = I_new_batch, I_batch
        R_batch, R_new_batch = R_new_batch, R_batch
        
        # Verificar si alguna simulación explota
        validas = cp.all((R_batch >= 0) & (R_batch <= 1), axis=(1, 2))
        simulaciones_validas &= validas
        
        # Si todas las simulaciones explotaron, terminar
        if not cp.any(simulaciones_validas):
            break
    
    # Calcular fitness para cada simulación en paralelo
    fitness_values = []
    
    # Crear máscaras para celdas quemadas (todo el batch de una vez)
    burnt_cells_sim_batch = cp.where(R_batch > 0.001, 1, 0)  # Shape: (batch_size, ny, nx)
    
    # Expandir burnt_cells para el batch
    burnt_cells_expanded = cp.broadcast_to(burnt_cells[cp.newaxis, :, :], (batch_size, ny, nx))
    
    # Calcular unión e intersección para todo el batch
    union_batch = cp.sum(burnt_cells_expanded | burnt_cells_sim_batch, axis=(1, 2))  # Shape: (batch_size,)
    interseccion_batch = cp.sum(burnt_cells_expanded & burnt_cells_sim_batch, axis=(1, 2))  # Shape: (batch_size,)
    
    # Calcular fitness para todo el batch
    burnt_cells_total = cp.sum(burnt_cells)
    fitness_batch = (union_batch - interseccion_batch) / burnt_cells_total
    
    # Procesar resultados
    for i in range(batch_size):
        if not simulaciones_validas[i]:
            fitness_values.append(float('inf'))
        else:
            fitness_values.append(float(fitness_batch[i]))
        
        # Información de debug
        params = parametros_batch[i]
        D, A, B, x, y = params[0], params[1], params[2], params[3], params[4]
        if simulaciones_validas[i]:
            print(f'Sim {i}: fitness={fitness_batch[i]:.4f}, D={D:.4f}, A={A:.4f}, B={B:.4f}, x={x}, y={y}')
            print(f'  Celdas quemadas: {burnt_cells_total}, Simuladas: {cp.sum(burnt_cells_sim_batch[i])}')
        else:
            print(f'Sim {i}: fitness=inf (simulación explotó), D={D:.4f}, A={A:.4f}, B={B:.4f}, x={x}, y={y}')

    return fitness_values