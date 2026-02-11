import numpy as np # type: ignore
import cupy as cp # type: ignore
import os

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
