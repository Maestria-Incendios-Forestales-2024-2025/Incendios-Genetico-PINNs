import cupy as cp
import numpy as np
from pathlib import Path

# Directorio raíz del proyecto
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Directorio de mapas
MAPAS_DIR = PROJECT_DIR / "mapas" / "mapas_steffen_martin"

ruta_mapas = {
    MAPAS_DIR / "ang_wind.asc",
    MAPAS_DIR / "speed_wind.asc",
    MAPAS_DIR / "asc_slope.asc",
    MAPAS_DIR / "asc_CIEFAP.asc",
    MAPAS_DIR / "asc_aspect.asc",
}

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
    vientod, vientov, pendiente, vegetacion, orientacion = datos

    # Parámetros derivados
    beta_veg = cp.where(vegetacion <= 2, 0, 0.1 * vegetacion)
    gamma = cp.where(vegetacion <= 2, 100, 0.1)

    vientod_rad = vientod * cp.pi / 180
    # Componentes cartesianas del viento:
    wx = -vientov * cp.sin(vientod_rad) * 1000  # Este = sin(ángulo desde Norte)
    wy = -vientov * cp.cos(vientod_rad) * 1000 # Norte = cos(ángulo desde Norte)

    slope = pendiente / 100.0 # pendiente en porcentaje
    orientacion_rad = orientacion * cp.pi / 180

    h_dx = -slope * cp.sin(orientacion_rad)
    h_dy = -slope * cp.cos(orientacion_rad)

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