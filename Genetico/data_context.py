from lectura_datos import leer_asc
import cupy as cp
import sys, os

# Agrega el directorio padre al path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mapas.io_mapas import ruta_mapas

class DataContext:
    def __init__(self):
        self.vientod = None
        self.vientov = None
        self.pendiente = None
        self.vegetacion = None
        self.orientacion = None
        self.wx = None
        self.wy = None
        self.h_dx = None
        self.h_dy = None
        self.ny = None
        self.nx = None
        
    def load(self):
        datos = [leer_asc(m) for m in ruta_mapas]
        vientod, vientov, pendiente, vegetacion, orientacion = datos

        vientod_rad = vientod * cp.pi / 180
        # Componentes cartesianas del viento:
        wx = -vientov * cp.sin(vientod_rad) * 1000  # Este = sin(ángulo desde Norte)
        wy = -vientov * cp.cos(vientod_rad) * 1000 # Norte = cos(ángulo desde Norte)

        slope = pendiente / 100.0 # pendiente en porcentaje
        orientacion_rad = orientacion * cp.pi / 180

        h_dx = -slope * cp.sin(orientacion_rad)
        h_dy = -slope * cp.cos(orientacion_rad)

        self.vientod = cp.flipud(vientod)
        self.vientov = cp.flipud(vientov)
        self.pendiente = cp.flipud(pendiente)
        self.vegetacion = cp.flipud(vegetacion)
        self.orientacion = cp.flipud(orientacion)
        self.wx = cp.flipud(wx)
        self.wy = cp.flipud(wy)
        self.h_dx = cp.flipud(h_dx)
        self.h_dy = cp.flipud(h_dy)
        self.ny = vientod.shape[0]
        self.nx = vientod.shape[1]

        return self