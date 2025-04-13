import cupy as cp # type: ignore

# Elementwise kernel
spread_kernel = cp.ElementwiseKernel(
    """
    float32 S, float32 I, float32 R, float32 dt, float32 d, float32 beta, float32 gamma, 
    float32 D, float32 wx, float32 wy, float32 h_dx, float32 h_dy, float32 A, float32 B,
    float32 I_top, float32 I_bottom, float32 I_left, float32 I_right
    """,
    """
    float32 S_new, float32 I_new, float32 R_new
    """,
    """
    S_new = S - dt * beta * I * S;

    float laplacian_I = (I_top + I_bottom + I_left + I_right - 4 * I);
    float s = D * dt / (d * d);

    float I_dx = ((A * wx + B * h_dx) > 0) ? (I - I_left) : (I_right - I);
    float I_dy = ((A * wy + B * h_dy) > 0) ? (I - I_bottom) : (I_top - I);

    I_new = I + dt * (beta * I * S - gamma * I) + s * laplacian_I 
            - dt / d * ((A * wx + B * h_dx) * I_dx + (A * wy + B * h_dy) * I_dy);

    R_new = R + dt * gamma * I;
    """,
    "spread_kernel"
)

# Versión hecha con elementwise kernel de cupy
def spread_infection(S, I, R, S_new, I_new, R_new, dt, d, beta, gamma, D, wx, wy, h_dx, h_dy, A, B):
    I_top = cp.roll(I, -1, axis=0)
    I_bottom = cp.roll(I, 1, axis=0)
    I_left = cp.roll(I, +1, axis=1)
    I_right = cp.roll(I, -1, axis=1)

    spread_kernel(S, I, R, dt, d, beta, gamma, D, wx, wy, h_dx, h_dy, A, B,
                  I_top, I_bottom, I_left, I_right,
                  S_new, I_new, R_new)

    # Aplicar condiciones de borde (esto se hace fuera del kernel)
    S_new[0, :] = S_new[-1, :] = S_new[:, 0] = S_new[:, -1] = 0
    I_new[0, :] = I_new[-1, :] = I_new[:, 0] = I_new[:, -1] = 0
    R_new[0, :] = R_new[-1, :] = R_new[:, 0] = R_new[:, -1] = 0

    # Celdas no combustibles
    no_fuel = (beta == 0)
    S_new[no_fuel] = 1
    I_new[no_fuel] = 0
    R_new[no_fuel] = 0

# Versión hecha completamente en cupy
def spread_infection_cupy(S, I, R, S_new, I_new, R_new, dt, d, beta, gamma, D, wx, wy, h_dx, h_dy, A, B):
    
    """

    La función actualiza el estado de una cuadrícula de vegetación susceptible (S), incendiándose (I) y quemada (R) 
    en función de varios parámetros que incluyen el tiempo, la velocidad del viento, la pendiente del terreno y 
    las constantes de propagación del fuego. Utiliza un esquema de diferencias finitas para modelar la difusión 
    del fuego y un esquema upwind para considerar la influencia del viento y la pendiente.

    Args:
      S: vegetación susceptible
      I: vegetación incendiándose
      R: vegetación quemada
      dt: paso temporal
      d: tamaño de celda
      beta: fracción de vegetación incéndiandose por hora
      gamma: fracción de vegetación incéndiandose por hora.
              1/gamma es el tiempo promedio del incendio
      wx: velocidad del viento en la dirección x
      wy: velocidad del viento en la dirección y
      h_dx: pendiente en la dirección x
      h_dy: pendiente en la dirección y
      A: constante adimensional que modifica la intensidad del viento
      B: constante con unidades de velocidad que modifica la intensidad de la altura

    Returns:
      S_new: nuevos árboles susceptibles
      I_new: nuevos árboles quemándose
      R_new: nuevos árboles quemados

    """

    # Actualización del estado S
    S_new[...] = S - dt * beta * I * S

    # Actualización del estado I
    I_laplacian = (cp.roll(I, 1, axis=0) + cp.roll(I, -1, axis=0) +
                  cp.roll(I, 1, axis=1) + cp.roll(I, -1, axis=1) - 4 * I )

    # Número de difusión
    s = D*dt/(d*d)

    # Esquema upwind para la altura
    I_dx = ((A*wx+B*h_dx)>0)*(I - cp.roll(I, 1, axis=1)) + ((A*wx+B*h_dx)<0)*(cp.roll(I, -1, axis=1) - I)
    I_dy = ((A*wy+B*h_dy)>0)*(I - cp.roll(I, 1, axis=0)) + ((A*wy+B*h_dy)<0)*(cp.roll(I, -1, axis=0) - I)

    I_new[...] = I + dt * (beta * I * S - gamma * I) + s * I_laplacian - dt/d * ((A*wx+B*h_dx) * I_dx + (A*wy+B*h_dy) * I_dy)

    # Actualización del estado R
    R_new[...] = R + dt * gamma * I

    # Condiciones de borde de Dirichlet
    S_new[0, :] = S_new[-1, :] = S_new[:, 0] = S_new[:, -1] = 0
    I_new[0, :] = I_new[-1, :] = I_new[:, 0] = I_new[:, -1] = 0
    R_new[0, :] = R_new[-1, :] = R_new[:, 0] = R_new[:, -1] = 0

    # Creamos una máscara para las celdas no combustibles
    no_fuel = (beta == 0)

    S_new[no_fuel] = 1
    I_new[no_fuel] = 0
    R_new[no_fuel] = 0


def courant(dt, D, A, B, d, wx, wy, h_dx, h_dy):
    
    '''
    Verifica la condición de estabilidad de Courant para un modelo de difusión y advección.
    
    Args:
      dt (float): Intervalo de tiempo.
      D (float): Coeficiente de difusión.
      A (float): Coeficiente relacionado con la velocidad del viento.
      B (float): Coeficiente relacionado con la pendiente del terreno.
      d (float): Tamaño de la celda en el mapa.
      wx (float): Componente x de la velocidad del viento.
      wy (float): Componente y de la velocidad del viento.
      h_dx (float): Derivada parcial de la altura del terreno en la dirección x.
      h_dy (float): Derivada parcial de la altura del terreno en la dirección y.
    
    Returns:
      bool: True si la condición de estabilidad de Courant se cumple, False en caso contrario.
    
    '''

    # Courant para difusión
    courant_difusion = d**2 / (2 * D)
    
    # Courant para advección
    velocidad = cp.sqrt((A * wx + B * h_dx)**2 + (A * wy + B * h_dy)**2)
    courant_advectivo = d / (cp.sqrt(2) * cp.max(velocidad))

    # Retorna un booleano indicando si la condición de estabilidad se cumple
    return dt < cp.minimum(courant_difusion, courant_advectivo)