from config import ruta_mapas
import numpy as np # type: ignore
import cupy as cp # type: ignore

def leer_asc(ruta):
    with open(ruta, 'r') as f:
        for i in range(6):  
            f.readline()
        data = np.loadtxt(f) 
        return cp.array(data, dtype=cp.float32) 
    
def preprocesar_datos():
    datos = [leer_asc(m) for m in ruta_mapas]
    vientod, vientov, pendiente, vegetacion, orientacion = datos

    # Parámetros derivados
    beta_veg = cp.where(vegetacion <= 2, 0, 0.1 * vegetacion)
    gamma = cp.where(vegetacion <= 2, 100, 0.1)

    wx = vientov * cp.cos(5/2 * cp.pi - vientod * cp.pi / 180) * 1000
    wy = - vientov * cp.sin(5/2 * cp.pi - vientod * cp.pi / 180) * 1000

    h_dx = cp.tan(pendiente * cp.pi / 180) * cp.cos(orientacion * cp.pi / 180 - cp.pi/2)
    h_dy = cp.tan(pendiente * cp.pi / 180) * cp.sin(orientacion * cp.pi / 180 - cp.pi/2)

    return {
        "vientod": vientod,
        "vientov": vientov,
        "pendiente": pendiente,
        "vegetacion": vegetacion,
        "orientacion": orientacion,
        "beta_veg": beta_veg,
        "gamma": gamma,
        "wx": wx,
        "wy": wy,
        "h_dx": h_dx,
        "h_dy": h_dy,
        "ny": vientod.shape[0],
        "nx": vientod.shape[1],
    }